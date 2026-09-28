"""
PySide6 + PyVista GUI viewer (AGENTS.md section 4.4 / Goal 4).

Workflow this GUI is the last step of:

    main_1.py  -- pick an HBM model + PPE, seat it (ICP/CPD), write a .feb
    (you)      -- solve the .feb in FEBio Studio / febio4.exe
    main_2.py  -- build the "GUI case" file from the solved outputs
    THIS FILE  -- pick that same HBM model + PPE from the dropdowns below,
                  and scrub through the real solved deformation

Dropdowns at the top select the HBM model and PPE; only combinations that
actually have a built GUI case (``postprocess.gui_case.discover_gui_cases``,
scanning ``cases_generated/*_gui_case.npz``) are offered -- no hardcoded
per-model/per-PPE case list to keep in sync.

The slider scrubs through the case's own real solved FEBio steps (linearly
interpolated between adjacent steps -- see ``app.gui_case_loader``), not a
synthetic loading curve: an earlier version of this app only had one fully
converged solved state to show, so it faked intermediate deformation with
a hand-tuned Ogden-rubber-inspired curve. Now that ``main_2.py`` keeps
every solved step, the real data can drive the slider directly.

Three-view layout, restored verbatim (by explicit user request) from the
project's original prototype (see git history / the pre-rewrite
``app/viewer_app.py``), just re-wired onto this pipeline's real
dropdown-selected case data instead of one hardcoded file set:

  * Main view (the whole render window): context body (translucent) +
    target skin region in a plain "normal tissue" color, geometry only, no
    scalar coloring or PPE -- free rotate/pan/zoom of the deforming region
    without clutter.
  * Top-left inset: the same target region, colored by displacement
    magnitude (fixed color scale from this case's own max solved
    displacement), frontal parallel-projection camera locked to the
    target patch's own outward normal, with a scalar bar.
  * Bottom-left inset: target region (plain) + the rigid PPE approaching
    it, oblique 3/4 perspective camera angled off the PPE's own approach
    axis (a straight-on camera would make the PPE's translation toward
    the body nearly invisible under parallel projection, since motion
    purely along the view direction produces no change in screen
    position).

Both insets are embedded viewports transplanted into the SAME render
window as the main view (not separate floating windows) via a throwaway
off-screen ``pv.Plotter`` per inset, whose renderer is lifted out and
added to the main window's own render window at a normalized viewport
rectangle. Because they all share the main window's single interactor,
VTK auto-routes mouse rotate/zoom to whichever viewport is under the
cursor -- each panel is independently spinnable/zoomable for free, no
extra interaction code needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv
from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

# Make the repo root importable regardless of how this file is launched --
# `python viewer_app.py` from the repo root (via the thin root launcher),
# `python -m app.viewer_app`, and (the case this fixes) `python
# viewer_app.py` run directly from *inside* app/ itself, where Python only
# puts app/'s own directory on sys.path, not its parent -- so `import
# app.xxx` would otherwise fail with "No module named 'app'".
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.gui_case_loader import GuiCase, displacement_at_time, gather_indices_for_target, load_gui_case
from app.hbm_reader import BodyMesh, TargetRegion, extract_target_region, load_body_mesh
from app.plate_render import PlateRenderData, plate_points_at_time
from config.hbm_models import discover_hbm_models
from config.sites import resolve_site
from fitting.body_axes import derive_torso_axes
from postprocess.export_state import build_fe_model_with_ppe, export_meshes
from postprocess.gui_case import GuiCaseInfo, discover_gui_cases

APP_ROOT = Path(__file__).resolve().parent.parent
HBM_ROOT = APP_ROOT / "HBM"
CACHE_DIR = APP_ROOT / "cache"
CASES_DIR = APP_ROOT / "cases_generated"

SLIDER_STEPS = 1000
MESH_COLOR = "#d9a679"  # plain "normal" tissue color for the main view
PLATE_COLOR = "#5a5f66"  # dark metallic gray for the rigid plate
DISPLACEMENT_CMAP = "YlOrRd"  # warm at 0, not the hard-to-see cold blue of turbo/viridis
INSET_VIEWPORT = (0.02, 0.60, 0.36, 0.98)  # (x0, y0, x1, y1) normalized, top-left
PLATE_INSET_VIEWPORT = (0.02, 0.02, 0.36, 0.40)  # bottom-left


def _quad_faces(conn: np.ndarray) -> np.ndarray:
    """Build a VTK/pyvista face-connectivity array from an (M, 4) index array."""
    count = conn.shape[0]
    faces = np.empty((count, 5), dtype=np.int64)
    faces[:, 0] = 4
    faces[:, 1:] = conn
    return faces.ravel()


def _format_hbm_display(key: str) -> str:
    """"F05_Standing" -> "F05 - Standing" for the dropdown. Purely
    cosmetic -- every actual lookup uses the combo's stored raw key
    (``QComboBox.currentData()``), never this display text.
    """
    return key.replace("_", " - ")


def _format_ppe_display(key: str) -> str:
    """"armored_plate" -> "Armored Plate" for the dropdown. Purely
    cosmetic -- every actual lookup uses the combo's stored raw key
    (``QComboBox.currentData()``), never this display text.
    """
    return key.replace("_", " ").title()


def _compute_frontal_camera(
    target_mesh: pv.PolyData, body_xyz: np.ndarray, up_axis: np.ndarray
) -> Tuple[List[float], List[float], List[float]]:
    """
    A camera looking straight on at the target patch's outward-facing
    side, framed on that patch alone, with "up" aligned to this HBM
    model's own derived anatomical up axis (``up_axis``, superior direction
    -- NOT necessarily +Z, and NOT necessarily the same for every model;
    see ``fitting.body_axes`` for why this is derived per-model rather than
    assumed fixed).
    """
    with_normals = target_mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
    )
    avg_normal = np.asarray(with_normals.cell_data["Normals"]).mean(axis=0)
    avg_normal /= np.linalg.norm(avg_normal)

    target_centroid = np.asarray(target_mesh.points).mean(axis=0)
    body_centroid = body_xyz.mean(axis=0)
    outward = target_centroid - body_centroid
    if np.dot(avg_normal, outward) < 0:
        avg_normal = -avg_normal

    # NOT "whichever axis has the largest extent, assumed positive" (the
    # previous approach): that guess is wrong exactly half the time and was
    # the actual cause of the body rendering upside-down by default on
    # every launch. Also NOT a single hardcoded axis shared by every model
    # (the cause of a later bug -- F05_Seated's plate seating on the
    # posterior side, since its own real up/anterior directions differ from
    # F05_Standing's) -- use this model's own derived axis instead.
    up = np.asarray(up_axis, dtype=np.float64).copy()
    up = up - np.dot(up, avg_normal) * avg_normal
    if np.linalg.norm(up) < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
    up /= np.linalg.norm(up)

    target_extents = target_mesh.points.max(axis=0) - target_mesh.points.min(axis=0)
    diagonal = float(np.linalg.norm(target_extents))
    position = target_centroid + avg_normal * diagonal * 2.5
    return position.tolist(), target_centroid.tolist(), up.tolist()


def _compute_plate_view_camera(
    target_mesh: pv.PolyData,
    approach_direction: np.ndarray,
    travel_mm: float,
    up_axis: np.ndarray,
) -> Tuple[List[float], List[float], List[float]]:
    """
    A 3/4-oblique view of the target patch, angled off the PPE's own
    approach axis. A straight-on view down that same axis (as used for
    the displacement inset) would make the PPE's translation toward
    the body nearly invisible -- especially under parallel projection,
    where motion purely along the view direction produces no change in
    screen position at all. Blending in a side component makes the
    closing distance clearly visible as on-screen motion.
    """
    target_centroid = np.asarray(target_mesh.points).mean(axis=0)

    # This HBM model's own derived anatomical up axis (NOT "whichever
    # global axis has the largest body extent, assumed positive" -- that
    # guess is wrong exactly half the time -- and NOT a single hardcoded
    # axis shared by every model, since different raw HBM exports are not
    # guaranteed to share a coordinate convention; see fitting.body_axes).
    up = np.asarray(up_axis, dtype=np.float64).copy()
    up = up - np.dot(up, approach_direction) * approach_direction
    if np.linalg.norm(up) < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
    up /= np.linalg.norm(up)

    side = np.cross(up, approach_direction)
    side /= np.linalg.norm(side)

    view_direction = approach_direction * 0.7 + side * 0.9
    view_direction /= np.linalg.norm(view_direction)

    target_extents = target_mesh.points.max(axis=0) - target_mesh.points.min(axis=0)
    diagonal = float(np.linalg.norm(target_extents))
    distance = diagonal * 2.0 + travel_mm * 1.5
    position = target_centroid + view_direction * distance
    return position.tolist(), target_centroid.tolist(), up.tolist()


def _combine_bounds(*bounds_list: Tuple[float, ...]) -> Tuple[float, ...]:
    """Union of several ``(xmin, xmax, ymin, ymax, zmin, zmax)`` bounds."""
    arr = np.asarray(bounds_list)
    result = np.empty(6)
    result[0::2] = arr[:, 0::2].min(axis=0)
    result[1::2] = arr[:, 1::2].max(axis=0)
    return tuple(result)


class DeformationViewer(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GUARDS PPE Viewer")
        self.resize(1280, 900)

        self._cases: List[GuiCaseInfo] = []
        self._current_case: Optional[GuiCase] = None
        self._current_plate: Optional[PlateRenderData] = None
        self._current_body: Optional[BodyMesh] = None
        self._current_target: Optional[TargetRegion] = None
        self._context_conn: Optional[np.ndarray] = None
        self._target_rest_xyz: Optional[np.ndarray] = None
        self._target_gather_idx: Optional[np.ndarray] = None
        self._case_max_disp: float = 1.0

        self._target_mesh: Optional[pv.PolyData] = None
        self.inset_mesh: Optional[pv.PolyData] = None
        self.plate_inset_skin_mesh: Optional[pv.PolyData] = None
        self.plate_mesh: Optional[pv.PolyData] = None

        self.inset_renderer = None
        self.plate_inset_renderer = None
        self._inset_helper_plotter: Optional[pv.Plotter] = None
        self._plate_inset_helper_plotter: Optional[pv.Plotter] = None

        self._build_ui()
        self._refresh_case_list()

    # ------------------------------------------------------------------
    # UI scaffolding
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        layout.addWidget(self._build_selector_bar())

        self.plotter = QtInteractor(central)
        layout.addWidget(self.plotter, stretch=1)

        layout.addWidget(self._build_controls())

        self.setCentralWidget(central)
        self.statusBar().showMessage("Select an HBM model and PPE above.")

    def _build_selector_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)

        row.addWidget(QtWidgets.QLabel("Human Body Model:"))
        self.hbm_combo = QtWidgets.QComboBox()
        self.hbm_combo.currentIndexChanged.connect(self._on_hbm_or_ppe_changed)
        row.addWidget(self.hbm_combo)

        row.addSpacing(24)

        row.addWidget(QtWidgets.QLabel("PPE:"))
        self.ppe_combo = QtWidgets.QComboBox()
        self.ppe_combo.currentIndexChanged.connect(self._on_hbm_or_ppe_changed)
        row.addWidget(self.ppe_combo)

        row.addStretch(1)

        self.export_button = QtWidgets.QPushButton("Export Current State")
        self.export_button.clicked.connect(self._on_export_clicked)
        row.addWidget(self.export_button)

        refresh_button = QtWidgets.QPushButton("Refresh case list")
        refresh_button.clicked.connect(self._refresh_case_list)
        row.addWidget(refresh_button)

        return bar

    def _build_controls(self) -> QtWidgets.QWidget:
        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QGridLayout(controls)

        self.load_label = QtWidgets.QLabel("Load / Closure: 0%")
        self.load_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.load_slider.setRange(0, SLIDER_STEPS)
        self.load_slider.setValue(0)
        self.load_slider.valueChanged.connect(self._on_slider_changed)

        controls_layout.addWidget(self.load_label, 0, 0)
        controls_layout.addWidget(self.load_slider, 0, 1)
        controls_layout.setColumnStretch(1, 1)

        # "Export Current State" progress, bottom-left -- hidden until an
        # export is actually running (see _on_export_clicked).
        self.export_progress_bar = QtWidgets.QProgressBar()
        self.export_progress_bar.setRange(0, 100)
        self.export_progress_bar.setValue(0)
        self.export_progress_bar.setMaximumWidth(240)
        self.export_progress_bar.setVisible(False)
        controls_layout.addWidget(self.export_progress_bar, 1, 0)

        return controls

    # ------------------------------------------------------------------
    # Case discovery / dropdown population
    # ------------------------------------------------------------------
    def _refresh_case_list(self) -> None:
        self._cases = discover_gui_cases(CASES_DIR)

        hbm_keys = sorted({c.hbm_model_key for c in self._cases})
        ppe_keys = sorted({c.ppe_key for c in self._cases})

        self._populate_combo(self.hbm_combo, hbm_keys, _format_hbm_display)
        self._populate_combo(self.ppe_combo, ppe_keys, _format_ppe_display)

        if not self._cases:
            self.statusBar().showMessage(
                f"No GUI cases found under {CASES_DIR} -- run main_1.py, solve the .feb, "
                "then main_2.py to build one."
            )
            return
        self._load_selected_case()

    @staticmethod
    def _populate_combo(combo: QtWidgets.QComboBox, keys: List[str], display_fn) -> None:
        """Repopulate ``combo``, showing ``display_fn(key)`` for each raw
        ``key`` while storing the raw key itself as each item's data (via
        ``currentData()``) -- so lookups elsewhere are never affected by
        how a key happens to be displayed.
        """
        previous_key = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for key in keys:
            combo.addItem(display_fn(key), key)
        if previous_key in keys:
            combo.setCurrentIndex(keys.index(previous_key))
        combo.blockSignals(False)

    def _on_hbm_or_ppe_changed(self, *_args) -> None:
        self._load_selected_case()

    def _find_case_info(self, hbm_model_key: str, ppe_key: str) -> Optional[GuiCaseInfo]:
        for info in self._cases:
            if info.hbm_model_key == hbm_model_key and info.ppe_key == ppe_key:
                return info
        return None

    # ------------------------------------------------------------------
    # Loading a case into the scene
    # ------------------------------------------------------------------
    def _load_selected_case(self) -> None:
        hbm_model_key = self.hbm_combo.currentData()
        ppe_key = self.ppe_combo.currentData()
        if not hbm_model_key or not ppe_key:
            return

        info = self._find_case_info(hbm_model_key, ppe_key)
        if info is None:
            self.statusBar().showMessage(
                f"No solved case available for HBM='{hbm_model_key}' + PPE='{ppe_key}'."
            )
            return

        self.statusBar().showMessage(f"Loading {info.path.name}...")
        QtWidgets.QApplication.processEvents()

        case = load_gui_case(info.path)

        self._current_case = case
        self._current_plate = self._load_plate_for_case(info)
        self._rebuild_scene(hbm_model_key, case.site_name)

    def _load_plate_for_case(self, info: GuiCaseInfo) -> PlateRenderData:
        # the gui_case npz already embeds plate geometry directly (written
        # by postprocess.gui_case.build_gui_case) -- load straight from it
        # rather than re-deriving a separate sidecar path.
        with np.load(info.path) as data:
            return PlateRenderData(
                nodes_rest=data["plate_nodes_rest"],
                boundary_faces=data["plate_boundary_faces"],
                push_direction=data["push_direction"],
                total_travel_mm=float(data["total_travel_mm"]),
                final_time=float(data["final_time"]),
            )

    def _clear_scene(self) -> None:
        """Remove any previously transplanted inset renderers before
        rebuilding (switching HBM model/PPE must not leak renderers or
        stack duplicate viewports on top of each other).
        """
        for renderer in (self.inset_renderer, self.plate_inset_renderer):
            if renderer is not None:
                self.plotter.ren_win.RemoveRenderer(renderer)
                try:
                    self.plotter.renderers._renderers.remove(renderer)
                except ValueError:
                    pass
        self.inset_renderer = None
        self.plate_inset_renderer = None
        self.plotter.clear()
        # clear() wipes the renderer's lights along with its actors -- with
        # zero lights left, VTK silently falls back to a single default
        # headlight (always exactly aligned with the view direction), which
        # is why the main view rendered completely flat/washed out
        # regardless of camera angle (confirmed: every real render, even
        # the very first, goes through this clear() path via
        # _refresh_case_list -> _load_selected_case -> _rebuild_scene, so
        # the light kit was never actually active). Re-enable the same
        # multi-light kit the (never-cleared) insets use, so real
        # Lambertian shading/contour is visible here too.
        self.plotter.enable_lightkit()

    def _rebuild_scene(self, hbm_model_key: str, site_name: str) -> None:
        models = discover_hbm_models(HBM_ROOT)
        if hbm_model_key not in models:
            self.statusBar().showMessage(f"HBM model '{hbm_model_key}' not found under {HBM_ROOT}.")
            return
        model = models[hbm_model_key]
        site = resolve_site(model, site_name)

        def _progress(msg: str) -> None:
            self.statusBar().showMessage(msg)
            QtWidgets.QApplication.processEvents()

        # This model's own derived anatomical up axis (NOT a single
        # hardcoded axis shared by every model -- different raw HBM
        # exports are not guaranteed to share a coordinate convention; see
        # fitting.body_axes module docstring for the real bug this fixes).
        # Cached under CACHE_DIR since deriving it means a fresh scan of
        # the ~200MB elements deck (~1 minute uncached).
        self._current_body_up_axis = derive_torso_axes(model, cache_dir=CACHE_DIR).up

        body = load_body_mesh(model, CACHE_DIR, progress=_progress)
        target = extract_target_region(body, site.shell_pids)

        self._current_body = body
        self._current_target = target
        self._target_rest_xyz = body.xyz[target.body_indices]

        context_mask = ~np.isin(body.shell_pid, list(site.shell_pids))
        self._context_conn = body.shell_conn[context_mask]

        # The solved case's own node set (case.node_ids) is the whole torso
        # region (skin + flesh); the rendered target region here is
        # skin-only, a reordered subset -- gather, don't assume alignment
        # (see gather_indices_for_target's docstring for the real bug this
        # was caught fixing: a straight (47544,3)+(11412,3) broadcast).
        gather_idx = gather_indices_for_target(self._current_case.node_ids, target.node_ids)
        if np.any(gather_idx < 0):
            missing = int(np.sum(gather_idx < 0))
            self.statusBar().showMessage(
                f"WARNING: {missing} of {len(target.node_ids)} target nodes have no solved "
                "displacement in this case; they will not move."
            )
        self._target_gather_idx = gather_idx

        # Fixed color-scale upper bound for the displacement inset, from
        # this case's own max solved displacement over the whole loading
        # history (not recomputed per frame) -- matches the original
        # prototype's behavior of a single clim set once at scene-build time.
        safe_idx = np.clip(gather_idx, 0, self._current_case.displacement.shape[1] - 1)
        gathered_disp = self._current_case.displacement[:, safe_idx, :]
        gathered_disp[:, gather_idx < 0, :] = 0.0
        self._case_max_disp = max(float(np.linalg.norm(gathered_disp, axis=2).max()), 1e-6)

        self._clear_scene()

        target_faces = _quad_faces(target.conn_local)

        # --- Main view: plain geometry only, no scalar coloring, no PPE ---
        self._add_context_body(self.plotter)
        self._target_mesh = pv.PolyData(self._target_rest_xyz.copy(), target_faces)
        self.plotter.add_mesh(
            self._target_mesh,
            color=MESH_COLOR,
            show_edges=False,
            name="target_skin_main",
        )

        self._build_inset(target_faces)
        self._build_plate_inset(target_faces)

        # Set the main view's camera LAST -- reset_camera()'s canonical
        # axis-aligned default happens to look almost exactly along this
        # target region's own average surface normal, a near-perpendicular
        # angle where the shared camera-relative "light kit" lands at
        # ~max diffuse everywhere (confirmed: rendered pixels exactly
        # matched the flat/unlit base color, no visible shading at all --
        # unlike either inset, which use a genuinely oblique camera and so
        # show real highlight-to-shadow gradient/contour). Reuse the same
        # oblique camera math as the plate inset below so the main view's
        # default look actually shows shape/deformation the same way, not
        # a flat wash of color -- still freely rotatable afterward.
        position, focal_point, up = _compute_plate_view_camera(
            self._target_mesh,
            -self._current_plate.push_direction,
            self._current_plate.total_travel_mm,
            self._current_body_up_axis,
        )
        self.plotter.camera_position = [position, focal_point, up]
        self.plotter.camera.SetParallelProjection(False)
        self.plotter.reset_camera(bounds=self._target_mesh.bounds)

        target_title = site_name
        self.statusBar().showMessage(
            f"{_format_hbm_display(hbm_model_key)} / {_format_ppe_display(self._current_case.ppe_key)}  |  "
            f"target: {target_title}"
        )

        self.load_slider.blockSignals(True)
        self.load_slider.setValue(0)
        self.load_slider.blockSignals(False)
        self._update_deformation()

    def _add_context_body(self, plotter) -> None:
        context_mesh = pv.PolyData(self._current_body.xyz, _quad_faces(self._context_conn))
        plotter.add_mesh(
            context_mesh, color="#cfcfcf", opacity=0.35, show_edges=False, name="context_body"
        )

    def _build_inset(self, target_faces: np.ndarray) -> None:
        """
        A truly embedded inset viewport in the top-left corner of the
        SAME render window (not a separate floating window), colored by
        displacement magnitude. Built via a throwaway off-screen Plotter
        so we get full add_mesh/scalar-bar support, then its renderer is
        transplanted into the main render window as a second, overlapping
        viewport. Because it shares the main window's single interactor,
        VTK automatically routes mouse rotate/zoom to whichever renderer
        is under the cursor -- so this box is independently spinnable and
        zoomable, with no extra interaction code needed.
        """
        helper = pv.Plotter(off_screen=True, shape=(1, 1))
        # Kept alive on self: the transplanted renderer only holds a
        # weakref back to its originating Plotter, so this one must stay
        # referenced for the lifetime of the viewer.
        self._inset_helper_plotter = helper

        context_mesh = pv.PolyData(self._current_body.xyz, _quad_faces(self._context_conn))
        helper.add_mesh(context_mesh, color="#cfcfcf", opacity=0.35, show_edges=False)

        self.inset_mesh = pv.PolyData(self._target_rest_xyz.copy(), target_faces)
        self.inset_mesh["displacement_mm"] = np.zeros(self._target_rest_xyz.shape[0])
        helper.add_mesh(
            self.inset_mesh,
            scalars="displacement_mm",
            cmap=DISPLACEMENT_CMAP,
            clim=[0.0, self._case_max_disp],
            show_edges=False,
            scalar_bar_args={
                "title": "Skin displacement (mm)",
                "fmt": "%.0f",
                "font_family": "arial",
                "vertical": False,
                "width": 0.6,
                "position_x": 0.2,
                "position_y": 0.04,
            },
        )

        position, focal_point, up = _compute_frontal_camera(
            self.inset_mesh, self._current_body.xyz, self._current_body_up_axis
        )
        helper.camera_position = [position, focal_point, up]
        helper.camera.SetParallelProjection(True)
        helper.reset_camera(bounds=self.inset_mesh.bounds)

        self.inset_renderer = helper.renderer
        helper.ren_win.RemoveRenderer(self.inset_renderer)
        self.inset_renderer.viewport = INSET_VIEWPORT

        self.plotter.ren_win.AddRenderer(self.inset_renderer)
        self.plotter.renderers._renderers.append(self.inset_renderer)
        self.plotter.render()

    def _build_plate_inset(self, target_faces: np.ndarray) -> None:
        """
        A second embedded inset (bottom-left), built the same way as the
        displacement inset above, but showing the rigid PPE approaching
        the torso instead -- plain colors, no displacement scalar field.
        See app/plate_render.py for how the PPE's geometry/motion is
        derived (a pure analytic translation, no per-node logging needed).
        """
        helper = pv.Plotter(off_screen=True, shape=(1, 1))
        self._plate_inset_helper_plotter = helper

        context_mesh = pv.PolyData(self._current_body.xyz, _quad_faces(self._context_conn))
        helper.add_mesh(context_mesh, color="#cfcfcf", opacity=0.35, show_edges=False)

        self.plate_inset_skin_mesh = pv.PolyData(self._target_rest_xyz.copy(), target_faces)
        helper.add_mesh(self.plate_inset_skin_mesh, color=MESH_COLOR, show_edges=False)

        plate = self._current_plate
        plate_faces = _quad_faces(plate.boundary_faces)
        self.plate_mesh = pv.PolyData(plate_points_at_time(plate, 0.0), plate_faces)
        helper.add_mesh(self.plate_mesh, color=PLATE_COLOR, show_edges=False)

        # _compute_plate_view_camera expects an OUTWARD approach direction
        # (as the original prototype's plate_model.retraction_direction
        # was) -- plate.push_direction is the INTO-body direction (see
        # app/plate_render.py's docstring), so negate it here.
        position, focal_point, up = _compute_plate_view_camera(
            self.plate_inset_skin_mesh,
            -plate.push_direction,
            plate.total_travel_mm,
            self._current_body_up_axis,
        )
        helper.camera_position = [position, focal_point, up]
        helper.camera.SetParallelProjection(False)
        helper.reset_camera(
            bounds=_combine_bounds(self.plate_inset_skin_mesh.bounds, self.plate_mesh.bounds)
        )

        self.plate_inset_renderer = helper.renderer
        helper.ren_win.RemoveRenderer(self.plate_inset_renderer)
        self.plate_inset_renderer.viewport = PLATE_INSET_VIEWPORT

        self.plotter.ren_win.AddRenderer(self.plate_inset_renderer)
        self.plotter.renderers._renderers.append(self.plate_inset_renderer)
        self.plotter.render()

    # ------------------------------------------------------------------
    # Slider -> deformation
    # ------------------------------------------------------------------
    def _on_slider_changed(self, *_args) -> None:
        self._update_deformation()

    def _current_load_fraction_and_time(self, case: GuiCase):
        """The slider's own ``(load_fraction, t)`` pair, shared by
        ``_update_deformation`` (drives the on-screen render) and the
        export handler (must export exactly what's currently on screen,
        not a snapped-to-nearest-solved-step approximation).
        """
        load_fraction = self.load_slider.value() / SLIDER_STEPS
        t_min, t_max = float(case.times[0]), float(case.times[-1])
        t = t_min + load_fraction * (t_max - t_min)
        return load_fraction, t

    def _update_deformation(self) -> None:
        case = self._current_case
        plate = self._current_plate
        if (
            case is None
            or plate is None
            or self._target_mesh is None
            or self._target_gather_idx is None
        ):
            return

        load_fraction, t = self._current_load_fraction_and_time(case)

        full_disp = displacement_at_time(case, t)
        safe_idx = np.clip(self._target_gather_idx, 0, len(full_disp) - 1)
        disp = full_disp[safe_idx].copy()
        disp[self._target_gather_idx < 0] = 0.0

        new_points = self._target_rest_xyz + disp
        disp_mag = np.linalg.norm(disp, axis=1)

        self._target_mesh.points = new_points

        self.inset_mesh.points = new_points
        self.inset_mesh["displacement_mm"] = disp_mag

        self.plate_inset_skin_mesh.points = new_points
        self.plate_mesh.points = plate_points_at_time(plate, t)

        self.plotter.render()

        self.load_label.setText(
            f"Load / Closure: {load_fraction * 100:.0f}%   max|u|={disp_mag.max():.1f}mm"
        )

    # ------------------------------------------------------------------
    # Export current state
    # ------------------------------------------------------------------
    def _make_export_progress_fn(self, base: float, weight: float):
        """A ``progress(message, fraction)`` callback (matching
        ``postprocess.export_state.ProgressFn``) that maps that function's
        own independent 0-1 ``fraction`` into this button's single overall
        0-100% bar, via ``base``/``weight`` (this phase's share of the
        whole action) -- so neither ``export_meshes`` nor
        ``build_fe_model_with_ppe`` needs to know the other exists, or how
        much of the overall bar it corresponds to.
        """

        def _progress(msg: str, fraction: float) -> None:
            overall_pct = round((base + fraction * weight) * 100)
            self.export_progress_bar.setValue(overall_pct)
            self.statusBar().showMessage(msg)
            QtWidgets.QApplication.processEvents()

        return _progress

    def _on_export_clicked(self) -> None:
        case = self._current_case
        plate = self._current_plate
        if case is None or plate is None or self._current_body is None:
            self.statusBar().showMessage("Select an HBM model and PPE with a loaded case first.")
            return

        models = discover_hbm_models(HBM_ROOT)
        model = models.get(case.hbm_model_key)
        if model is None:
            self.statusBar().showMessage(f"HBM model '{case.hbm_model_key}' not found under {HBM_ROOT}.")
            return
        site = resolve_site(model, case.site_name)

        load_fraction, t = self._current_load_fraction_and_time(case)
        pct = round(load_fraction * 100)

        out_dir = CASES_DIR / f"{case.hbm_model_key}_{case.ppe_key}_export_{pct}pct"

        self.export_button.setEnabled(False)
        self.export_progress_bar.setValue(0)
        self.export_progress_bar.setVisible(True)
        try:
            self.statusBar().showMessage(f"Exporting meshes at {pct}% load to {out_dir}...")
            QtWidgets.QApplication.processEvents()

            # Phase 1 (export_meshes) is roughly a third of the total time,
            # phase 2 (the full-body streaming patch) the rest -- see the
            # AGENTS.md writeup on this feature for the measured split.
            export_meshes(
                model, site, case, plate, case.ppe_key, t, out_dir,
                progress=self._make_export_progress_fn(0.0, 0.35),
            )
            self.statusBar().showMessage(f"Meshes Exported -- {out_dir}")
            QtWidgets.QApplication.processEvents()

            fe_model_path = out_dir / f"{case.hbm_model_key}_with_ppe.k"
            build_fe_model_with_ppe(
                model, self._current_body, case, plate, case.ppe_key, t, fe_model_path,
                progress=self._make_export_progress_fn(0.35, 0.65),
            )
            self.export_progress_bar.setValue(100)
            self.statusBar().showMessage(f"FE Model Built with PPE -- {fe_model_path}")
        except Exception as exc:  # noqa: BLE001 -- surface any failure to the user, don't crash the GUI
            self.statusBar().showMessage(f"Export failed: {exc}")
            raise
        finally:
            self.export_button.setEnabled(True)
            self.export_progress_bar.setVisible(False)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    viewer = DeformationViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
