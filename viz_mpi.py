"""Distributed visualization of f(x,y,z) = 2550 sin(50x) sin(50y) cos(50z).

Part D of the Visualization lecture activity (Physics 305 HPC).

Run with, e.g.
    mpirun -n 4 python viz_mpi.py --n 256 --out results/p4

The pipeline follows the activity's diagram, one stage per timer:

    generate   each rank fills its own z-slab of the grid (stands in for "read")
    gradient   each rank differentiates its slab            (parallel processing)
    contour    each rank extracts isosurface triangles      (parallel processing)
    stream     each rank traces streamlines of grad f       (parallel processing)
    render_*   each rank draws its own geometry / volume    (parallel rendering)
    composite  partial images are gathered and merged       (image compositing)

Compositing is sort-last, as in ParaView/IceT: every rank renders a full-size
image with the same camera, and the images are merged per pixel -- by depth for
opaque geometry, and back-to-front "over" blending for the volume rendering.
"""
import argparse
import json
import os

# One thread per rank, so that speedup comes from MPI ranks only.
os.environ.setdefault("VTK_SMP_BACKEND_IN_USE", "Sequential")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from mpi4py import MPI
import vtk
from vtk.util import numpy_support as ns

AMP, K = 2550.0, 50.0
BOX = np.pi / K  # edge of one nodal cell of f; a gradient line never leaves its cell

comm = MPI.COMM_WORLD
rank, nproc = comm.Get_rank(), comm.Get_size()


def f_analytic(x, y, z):
    return AMP * np.sin(K * x) * np.sin(K * y) * np.cos(K * z)


def grad_analytic(p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return AMP * K * np.column_stack([
        np.cos(K * x) * np.sin(K * y) * np.cos(K * z),
        np.sin(K * x) * np.cos(K * y) * np.cos(K * z),
        -np.sin(K * x) * np.sin(K * y) * np.sin(K * z),
    ])


class Timer:
    """Wall-clock per stage. Barriers make every rank start a stage together,
    so the stage time reported is that of the slowest rank."""

    def __init__(self):
        self.t = {}

    def run(self, name, fn):
        comm.Barrier()
        t0 = MPI.Wtime()
        out = fn()
        local = MPI.Wtime() - t0
        self.t[name] = self.t.get(name, 0.0) + comm.allreduce(local, op=MPI.MAX)
        return out


def make_image(arr, origin, h):
    """Wrap a (nz, ny, nx) or (nz, ny, nx, 3) numpy block as vtkImageData."""
    nz, ny, nx = arr.shape[:3]
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetOrigin(*origin)
    img.SetSpacing(h, h, h)
    return img


def add_array(img, arr, name, vectors=False):
    flat = arr.reshape(-1, 3) if vectors else arr.ravel()
    v = ns.numpy_to_vtk(flat, deep=True)
    v.SetName(name)
    if vectors:
        img.GetPointData().SetVectors(v)
    else:
        img.GetPointData().SetScalars(v)


def new_window(size):
    ren = vtk.vtkRenderer()
    ren.SetBackground(0, 0, 0)
    ren.SetBackgroundAlpha(0.0)
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetAlphaBitPlanes(1)
    win.SetMultiSamples(0)
    win.SetSize(size, size)
    win.AddRenderer(ren)
    return ren, win


def set_camera(ren, L):
    """Same camera on every rank: explicit position *and* clipping range, so
    that depth values from different ranks are directly comparable."""
    cam = ren.GetActiveCamera()
    c = L / 2
    cam.SetFocalPoint(c, c, c)
    cam.SetPosition(c + 2.1 * L, c + 1.7 * L, c + 2.6 * L)
    cam.SetViewUp(0, 1, 0)
    cam.SetViewAngle(30)
    cam.SetClippingRange(1.0 * L, 8.0 * L)
    return cam


def grab(win, size, depth=True):
    win.Render()
    rgba = vtk.vtkUnsignedCharArray()
    win.GetRGBACharPixelData(0, 0, size - 1, size - 1, 0, rgba)
    rgba = ns.vtk_to_numpy(rgba).reshape(size, size, 4).copy()
    z = None
    if depth:
        zb = vtk.vtkFloatArray()
        win.GetZbufferData(0, 0, size - 1, size - 1, zb)
        z = ns.vtk_to_numpy(zb).reshape(size, size).copy()
    return rgba, z


def composite_depth(parts):
    """Opaque geometry: per pixel, keep the fragment nearest the camera."""
    rgba, z = parts[0]
    rgba, z = rgba.copy(), z.copy()
    for r2, z2 in parts[1:]:
        m = z2 < z
        rgba[m] = r2[m]
        z[m] = z2[m]
    return rgba


def composite_over(parts, order):
    """Translucent volume: blend partial images back to front.
    Colours are premultiplied by alpha (rendered over a black, alpha=0 bg)."""
    acc = np.zeros(parts[0].shape, dtype=np.float32)
    for i in order:
        src = parts[i].astype(np.float32) / 255.0
        acc = src + acc * (1.0 - src[..., 3:4])
    return (np.clip(acc, 0, 1) * 255).astype(np.uint8)


def write_png(rgba, path, bg=(1.0, 1.0, 1.0)):
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    out = rgb + np.array(bg, dtype=np.float32) * (1 - a)   # colours are premultiplied
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)[::-1]  # VTK rows start at the bottom
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    mpimg.imsave(path, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="grid points per axis")
    ap.add_argument("--length", type=float, default=1.0, help="domain is [0, length]^3")
    ap.add_argument("--size", type=int, default=800, help="image is size x size pixels")
    ap.add_argument("--seeds", type=int, default=8000, help="streamline seed points")
    ap.add_argument("--iso", type=float, nargs="+", default=[-1275.0, 1275.0])
    ap.add_argument("--out", default="results/run")
    ap.add_argument("--no-images", action="store_true", help="time only, skip PNG output")
    a = ap.parse_args()

    N, L, S = a.n, a.length, a.size
    h = L / (N - 1)
    tm = Timer()
    t_total0 = MPI.Wtime()

    # ---- partition: split the N-1 cell layers along z into nproc slabs --------
    cuts = np.linspace(0, N - 1, nproc + 1).round().astype(int)
    k0, k1 = cuts[rank], cuts[rank + 1]      # owned points k0..k1 (shared faces)
    # Ghost layers: enough for a gradient line (confined to one nodal cell of
    # edge pi/50) seeded in the owned slab to be traced without leaving the block.
    G = int(np.ceil(BOX / h)) + 2
    g0, g1 = max(k0 - G, 0), min(k1 + G, N - 1)

    ax = (np.arange(N) * h).astype(np.float64)

    # ---- generate (stand-in for reading this rank's piece from disk) ----------
    def generate():
        z = ax[g0:g1 + 1]
        return (AMP * np.cos(K * z)[:, None, None] * np.sin(K * ax)[None, :, None]
                * np.sin(K * ax)[None, None, :]).astype(np.float32)
    f_pad = tm.run("generate", generate)
    f_own = f_pad[k0 - g0:k1 - g0 + 1]
    own_img = make_image(f_own, (0, 0, k0 * h), h)
    add_array(own_img, f_own, "f")

    # ---- processing 1: gradient (needs neighbours -> ghost layers) ------------
    def gradient():
        gz, gy, gx = np.gradient(f_pad, h)
        return np.stack([gx, gy, gz], axis=-1)
    grad = tm.run("gradient", gradient)
    pad_img = make_image(f_pad, (0, 0, g0 * h), h)
    add_array(pad_img, f_pad, "f")
    add_array(pad_img, grad, "grad_f", vectors=True)
    del grad

    # ---- processing 2: isosurfaces --------------------------------------------
    def contour():
        cf = vtk.vtkFlyingEdges3D()
        cf.SetInputData(own_img)
        for i, v in enumerate(a.iso):
            cf.SetValue(i, v)
        cf.ComputeNormalsOff()
        cf.ComputeScalarsOn()
        cf.Update()
        pd = cf.GetOutput()
        if pd.GetNumberOfPoints():
            # exact normals from the analytic gradient: no seams between slabs
            # (minus sign: VTK's contour winding faces down-gradient)
            g = -grad_analytic(ns.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64))
            g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30)
            nrm = ns.numpy_to_vtk(g.astype(np.float32), deep=True)
            nrm.SetName("Normals")
            pd.GetPointData().SetNormals(nrm)
        return pd
    iso = tm.run("contour", contour)

    # ---- processing 3: streamlines of grad f ----------------------------------
    def stream():
        rng = np.random.default_rng(12345)           # same seeds on every rank
        pts = rng.uniform(0.02 * L, 0.98 * L, size=(a.seeds, 3))
        zlo, zhi = k0 * h, k1 * h
        mine = pts[(pts[:, 2] >= zlo) & ((pts[:, 2] < zhi) | (rank == nproc - 1))]
        seeds = vtk.vtkPolyData()
        vp = vtk.vtkPoints()
        vp.SetData(ns.numpy_to_vtk(mine, deep=True))
        seeds.SetPoints(vp)
        st = vtk.vtkStreamTracer()
        st.SetInputData(pad_img)
        st.SetSourceData(seeds)
        st.SetIntegratorTypeToRungeKutta45()
        st.SetIntegrationDirectionToForward()         # uphill, toward the local maximum
        st.SetMaximumPropagation(4 * BOX)
        st.SetInitialIntegrationStep(0.2)             # in cell lengths
        st.SetMaximumIntegrationStep(0.25)
        st.SetTerminalSpeed(2e-2 * AMP * K)      # stop once the line reaches the maximum
        st.SetMaximumNumberOfSteps(2000)
        st.Update()
        return st.GetOutput()
    lines = tm.run("stream", stream)

    # ---- rendering: every rank draws only its own piece -----------------------
    lut = vtk.vtkColorTransferFunction()
    lut.SetColorSpaceToDiverging()
    lut.AddRGBPoint(-AMP, 0.23, 0.30, 0.75)
    lut.AddRGBPoint(0.0, 0.87, 0.87, 0.87)
    lut.AddRGBPoint(AMP, 0.71, 0.02, 0.15)

    part = {}

    def render_iso():
        ren, win = new_window(S)
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(iso)
        m.SetLookupTable(lut)
        m.UseLookupTableScalarRangeOn()
        act = vtk.vtkActor()
        act.SetMapper(m)
        ren.AddActor(act)
        set_camera(ren, L)
        part["iso"] = grab(win, S)
        win.Finalize()
    tm.run("render_iso", render_iso)

    def render_stream():
        ren, win = new_window(S)
        m = vtk.vtkPolyDataMapper()
        m.SetInputData(lines)
        m.SetLookupTable(lut)
        m.UseLookupTableScalarRangeOn()
        m.SetScalarModeToUsePointFieldData()
        m.SelectColorArray("f")
        act = vtk.vtkActor()
        act.SetMapper(m)
        act.GetProperty().SetLineWidth(1.5)
        ren.AddActor(act)
        set_camera(ren, L)
        part["stream"] = grab(win, S)
        win.Finalize()
    tm.run("render_stream", render_stream)

    def render_volume():
        # software ray casting on one thread, so any speedup comes from ranks
        ren, win = new_window(S)
        vm = vtk.vtkFixedPointVolumeRayCastMapper()
        vm.SetInputData(own_img)
        vm.SetNumberOfThreads(1)
        vm.SetAutoAdjustSampleDistances(0)
        vm.SetSampleDistance(h)
        vm.SetImageSampleDistance(1.0)
        op = vtk.vtkPiecewiseFunction()
        for v, o in [(-AMP, 0.9), (-0.6 * AMP, 0.25), (-0.3 * AMP, 0.0),
                     (0.3 * AMP, 0.0), (0.6 * AMP, 0.25), (AMP, 0.9)]:
            op.AddPoint(v, o)
        prop = vtk.vtkVolumeProperty()
        prop.SetColor(lut)
        prop.SetScalarOpacity(op)
        prop.SetScalarOpacityUnitDistance(BOX / 4)
        prop.SetInterpolationTypeToLinear()
        prop.ShadeOff()
        vol = vtk.vtkVolume()
        vol.SetMapper(vm)
        vol.SetProperty(prop)
        ren.AddVolume(vol)
        cam = set_camera(ren, L)
        part["volume"] = grab(win, S, depth=False)[0]
        part["cam_z"] = cam.GetPosition()[2]
        win.Finalize()
    tm.run("render_volume", render_volume)

    # ---- compositing: gather partial images on rank 0 and merge ----------------
    def composite():
        mid = 0.5 * (k0 + k1) * h
        gathered = comm.gather((part["iso"], part["stream"], part["volume"],
                                abs(mid - part["cam_z"])), root=0)
        if rank != 0:
            return None
        order = np.argsort([-g[3] for g in gathered])        # farthest slab first
        return {"iso": composite_depth([g[0] for g in gathered]),
                "stream": composite_depth([g[1] for g in gathered]),
                "volume": composite_over([g[2] for g in gathered], order)}
    final = tm.run("composite", composite)

    counts = comm.gather((iso.GetNumberOfPolys(), lines.GetNumberOfLines(),
                          lines.GetNumberOfPoints(), f_pad.size, f_own.size), root=0)
    total = MPI.Wtime() - t_total0

    if rank == 0:
        os.makedirs(a.out, exist_ok=True)
        c = np.array(counts)
        t = dict(tm.t)
        t["processing"] = t["gradient"] + t["contour"] + t["stream"]
        t["render"] = t["render_iso"] + t["render_stream"] + t["render_volume"]
        t["pipeline"] = sum(tm.t.values())
        res = {
            "ranks": nproc, "n": N, "length": L, "image_size": S, "ghost_layers": G,
            "times": t, "wall_including_setup": total,
            "triangles": int(c[:, 0].sum()), "streamlines": int(c[:, 1].sum()),
            "streamline_points": int(c[:, 2].sum()),
            "points_with_ghosts": int(c[:, 3].sum()), "points_owned": int(c[:, 4].sum()),
        }
        with open(os.path.join(a.out, "timing.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        if not a.no_images:
            for name in ("iso", "stream", "volume"):
                write_png(final[name], os.path.join(a.out, f"{name}.png"))
        print(json.dumps(res))


if __name__ == "__main__":
    main()
