#!/usr//bin/env python3
#
# Simulates the response of a memristor array to event-camera data.
# This script supports two primary simulation schemes:
#
# Scheme 1 (version=1): Event-count thresholding in discrete time windows.
# Scheme 2 (version=2): DC bias with event-triggered voltage overlays.
#   - Polarity mode (--polarity) controls whether ON/OFF events map to two
#     independent memristor arrays ('split') or a single array ('magnitude').
#
# Off-polarity events are handled as p=0.

import argparse, h5py, numpy as np, cv2, json, gzip
from pathlib import Path
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
# 0. Global Parameters
# ────────────────────────────────────────────────────────────────────────────────
PARAMS = dict(alphaoff=1, alphaon=1,
              voff=-0.2, von=0.1,
              koff=51.03, kon=-2.91,
              son=0.2,  soff=0.8,
              bon=-5.12, boff=3.10,
              Ron=163_305, Roff=2_104_377,
              won=1, woff=0,
              wini=0.5)

# Integration time-step, matching prior MATLAB implementations.
DT = 5e-4      # [s]  = 0.5 ms

# Internal simulation knobs
THETA_EVENTS = 1     # Scheme 1: Min event count in a window to trigger a pulse.
REFRACTORY_US = 800  # Scheme 2: Per-pixel refractory period (µs) to limit update rate.


# ────────────────────────────────────────────────────────────────────────────────
# 1. Memristor Model
# ────────────────────────────────────────────────────────────────────────────────
def update_state(w, V, p=PARAMS, dt=DT):
    """Vectorized state update for the memristor array."""
    dwdt = np.zeros_like(w, dtype=np.float32)

    mask_off = V < p['voff']
    mask_on  = V > p['von']

    if mask_off.any():
        dwdt[mask_off] = (p['koff']
                          * (V[mask_off] / p['voff'] - 1) ** p['alphaoff']
                          * (1 - w[mask_off] * p['soff']) ** p['boff'])
    if mask_on.any():
        dwdt[mask_on]  = (p['kon']
                          * (V[mask_on]  / p['von']  - 1) ** p['alphaon']
                          * (1 - w[mask_on]  * p['son'])  ** p['bon'])

    w_new = np.clip(w + dwdt * dt, 0.0, 1.0)
    return w_new


def resistance_exp(w, p=PARAMS):
    """Maps state `w` (0..1) to resistance using an exponential curve."""
    lam = np.log(p['Roff'] / p['Ron'])
    return p['Ron'] / np.exp(-lam * (1.0 - w))


# ────────────────────────────────────────────────────────────────────────────────
# 2. Event Data I/O and Utilities
# ────────────────────────────────────────────────────────────────────────────────
def load_events(h5_path):
    """Loads event data from an HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        evs = f['/CD/events']
        x, y, p, t = evs['x'][:], evs['y'][:], evs['p'][:].astype(int), evs['t'][:]
    H, W = y.max() + 1, x.max() + 1
    return x, y, p, t, H, W


def slice_indices(t, slice_us):
    """Yields index slices corresponding to fixed-duration time windows."""
    bounds = np.arange(t[0], t[-1] + slice_us, slice_us, dtype=t.dtype)
    idx = np.searchsorted(t, bounds)
    for i in range(len(idx) - 1):
        yield slice(idx[i], idx[i + 1])


def write_video(frames, out_path, fps):
    """Writes a list of numpy arrays to an MP4 video file."""
    if not frames:
        return
    H, W = frames[0].shape
    vw = cv2.VideoWriter(str(out_path),
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps, (W, H), isColor=False)
    for f in frames:
        img = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vw.write(img)
    vw.release()


def bincount_2d(x, y, H, W):
    """Performs a fast 2D histogram of event locations."""
    lin = (y.astype(np.int64) * W + x.astype(np.int64))
    bc = np.bincount(lin, minlength=H*W).reshape(H, W)
    return bc.astype(np.int32)

# ────────────────────────────────────────────────────────────────────────────────
# 2b. Synthetic Data Generator
# ────────────────────────────────────────────────────────────────────────────────
def generate_synthetic_events(H=240, W=320,
                              box_h=50, box_w=50,
                              speed_pps=300,
                              duration_s=1.5):
    """
    Generates a synthetic event stream of a box moving from left to right.

    The box is white (1) on a black (0) background. ON events (+1) are
    generated at the leading edge and OFF events (-1) at the trailing edge.

    Returns: tuple of (x, y, p, t_us) numpy arrays for the events.
    """
    events = []
    t_step_us = int(DT * 1_000_000)
    duration_us = int(duration_s * 1_000_000)
    box_y_start = (H - box_h) // 2

    prev_frame = np.zeros((H, W), dtype=np.uint8)

    for t_us in range(0, duration_us, t_step_us):
        t_s = t_us / 1_000_000
        curr_frame = np.zeros((H, W), dtype=np.uint8)
        box_x_start = int(t_s * speed_pps)
        box_x_end = box_x_start + box_w

        if box_x_start < W and box_x_end > 0:
            x_start_clip = max(0, box_x_start)
            x_end_clip = min(W, box_x_end)
            curr_frame[box_y_start : box_y_start + box_h,
                       x_start_clip:x_end_clip] = 1

        diff = curr_frame.astype(np.int8) - prev_frame.astype(np.int8)

        on_y, on_x = np.where(diff == 1)
        for y, x in zip(on_y, on_x):
            events.append((x, y, 1, t_us))

        off_y, off_x = np.where(diff == -1)
        for y, x in zip(off_y, off_x):
            events.append((x, y, -1, t_us))

        prev_frame = curr_frame

    if not events:
        return (np.array([], dtype=int), np.array([], dtype=int),
                np.array([], dtype=int), np.array([], dtype=int))

    events.sort(key=lambda e: e[3])
    x, y, p, t = zip(*events)
    return (np.array(x), np.array(y), np.array(p), np.array(t))


# ────────────────────────────────────────────────────────────────────────────────
# 3. Simulation Schemes
# ────────────────────────────────────────────────────────────────────────────────
def simulate(h5_path: Path,
             version: int = 1,
             slice_us: int = 1_000,
             active_v: float = -8.0,
             silent_v: float =  0.0,
             save_video: bool = True,
             polarity: str = 'split'):

    assert version in (1, 2), "version must be 1 or 2"
    assert polarity in ('split','magnitude'), "polarity must be 'split' or 'magnitude'"

    x, y, pol, t_us, H, W = load_events(h5_path)
    nslices   = sum(1 for _ in slice_indices(t_us, slice_us))
    fps       = 1_000_000 / slice_us

    # Allocate memristor state array(s)
    w = PARAMS['wini'] * np.ones((H, W), dtype=np.float32)
    res_hist = []
    vframes  = []

    # Only save every Nth frame to limit memory usage with large datasets
    save_every_n = max(1, nslices // 100)  # Aim for ~100 saved frames
    slice_counter = 0

    if version == 2:
        use_two = (polarity == 'split')
        if use_two:
            # A second, independent array for OFF events
            w_b        = w.copy()
            res_hist_b = []
            vframes_b  = []
            # Per-pixel refractory trackers (timestamps in µs)
            next_ok_on  = np.zeros((H, W), dtype=np.int64)
            next_ok_off = np.zeros((H, W), dtype=np.int64)
        else:
            w_b        = None
            res_hist_b = []
            vframes_b  = []
            next_ok    = np.zeros((H, W), dtype=np.int64)

    for sl in tqdm(slice_indices(t_us, slice_us),
                   total=nslices, desc=f'Version {version} ({polarity})'):

        # ────────────────────────── Scheme 1: Boxcar Window ──────────────────────────
        if version == 1:
            # Apply silent voltage by default
            V = np.full((H, W), silent_v, dtype=np.float32)

            # If events occurred, find pixels exceeding the event count threshold
            if sl.stop > sl.start:
                event_counts = bincount_2d(x[sl], y[sl], H, W)
                active_mask = (event_counts >= THETA_EVENTS)
                if active_mask.any():
                    V[active_mask] = active_v

            # Perform a single state update for the entire window
            w = update_state(w, V)

            if slice_counter % save_every_n == 0:
                res_hist.append(resistance_exp(w))
                if save_video:
                    vframes.append(V)

            slice_counter += 1
            continue

        # ─────────────────────── Scheme 2: DC Bias + Event Overlay ───────────────────────
        V_bias = float(silent_v)
        Va = np.full((H, W), V_bias, dtype=np.float32)
        if use_two:
            Vb = np.full((H, W), V_bias, dtype=np.float32)

        if sl.stop > sl.start:
            if use_two:
                # Apply ON-event overlays to array 'A'
                mask_on = (pol[sl] ==  1)
                if mask_on.any():
                    xs, ys = x[sl][mask_on], y[sl][mask_on]
                    # Respect per-pixel refractory period
                    ok = (next_ok_on[ys, xs] <= t_us[sl.start])
                    if np.any(ok):
                        xs_ok, ys_ok = xs[ok], ys[ok]
                        Va[ys_ok, xs_ok] += active_v
                        next_ok_on[ys_ok, xs_ok] = t_us[sl.stop-1] + REFRACTORY_US

                # Apply OFF-event overlays to array 'B'
                mask_off = (pol[sl] == 0) # Use p=0 for OFF events
                if mask_off.any():
                    xs, ys = x[sl][mask_off], y[sl][mask_off]
                    ok = (next_ok_off[ys, xs] <= t_us[sl.start])
                    if np.any(ok):
                        xs_ok, ys_ok = xs[ok], ys[ok]
                        Vb[ys_ok, xs_ok] += active_v
                        next_ok_off[ys_ok, xs_ok] = t_us[sl.stop-1] + REFRACTORY_US
            else:
                # Magnitude mode: all events trigger an overlay on array 'A'
                xs, ys = x[sl], y[sl]
                if xs.size:
                    # Find unique pixels with events in this slice
                    uniq = np.unique(ys.astype(np.int64)*W + xs.astype(np.int64))
                    uy, ux = np.divmod(uniq, W)
                    ok = (next_ok[uy, ux] <= t_us[sl.start])
                    if np.any(ok):
                        uy_ok, ux_ok = uy[ok], ux[ok]
                        Va[uy_ok, ux_ok] += active_v
                        next_ok[uy_ok, ux_ok] = t_us[sl.stop-1] + REFRACTORY_US

        # Update state arrays based on the computed voltage maps
        w = update_state(w, Va)
        if use_two:
            w_b = update_state(w_b, Vb)

        # Save state periodically
        if slice_counter % save_every_n == 0:
            res_hist.append(resistance_exp(w))
            if save_video:
                vframes.append(Va)
            if use_two:
                res_hist_b.append(resistance_exp(w_b))
                if save_video:
                    vframes_b.append(Vb)

        slice_counter += 1

    # ── Save results ──────────────────────────────────────────────────────────
    out_npz = h5_path.with_suffix(f'.V{version}.npz')
    np.savez_compressed(out_npz,
                        w_final=w,
                        resistances=np.array(res_hist, dtype=np.float32))
    if version == 2:
        out_npz_b = h5_path.with_suffix('.V2_b.npz')
        if use_two:
            np.savez_compressed(out_npz_b,
                                w_final=w_b,
                                resistances=np.array(res_hist_b, dtype=np.float32))
        else:
            # Create an empty file for pipeline compatibility in magnitude mode
            np.savez_compressed(out_npz_b,
                                w_final=np.array([]),
                                resistances=np.array([]))

    # ── Optional MP4 preview ─────────────────────────────────────────────────
    if save_video:
        vid_path = h5_path.with_suffix(f'.V{version}.mp4')
        write_video(vframes, vid_path, fps)
        if version == 2:
            write_video(vframes_b,
                        h5_path.with_suffix('.V2_b.mp4'), fps)

    # ── Persist simulation metadata ──────────────────────────────────────────
    meta = dict(version=version, slice_us=slice_us, fps=fps,
                params=PARAMS, dt=DT,
                scheme='boxcar' if version==1 else 'dc_bias_overlay',
                polarity=polarity if version==2 else None,
                theta_events=THETA_EVENTS if version==1 else None,
                refractory_us=REFRACTORY_US if version==2 else None,
                event_file=str(h5_path))
    with gzip.open(h5_path.with_suffix(f'.V{version}.json.gz'), 'wt') as fp:
        json.dump(meta, fp, indent=2)

    print(f'✔︎ Simulation finished. Results → {out_npz}')
    if version == 2:
        print(f'                          and {out_npz_b}')
    if save_video:
        print(f'                          video  → {vid_path}')


# ────────────────────────────────────────────────────────────────────────────────
# 4.  Command-line interface
# ────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Event-camera and memristor array simulator.")
    ap.add_argument('--h5', default='driving_data.hdf5',
                    help='HDF5 file with /CD/events (default: driving_data.hdf5)')
    ap.add_argument('--version', type=int, choices=[1, 2], default=1,
                    help='1 = Scheme 1 (boxcar); 2 = Scheme 2 (DC bias + overlay)')
    ap.add_argument('--slice_us', type=int, default=1_000,
                    help='Time window duration in microseconds (default: 1000)')
    ap.add_argument('--active_v', type=float, default=-6.0,
                    help='Voltage for active pixels or event overlays')
    ap.add_argument('--silent_v', type=float, default= 0.0,
                    help='Voltage for inactive pixels (Scheme 1) or DC bias (Scheme 2)')
    ap.add_argument('--polarity', choices=['split','magnitude'], default='split',
                    help='Scheme 2 only: use two arrays (split) or one array (magnitude)')
    ap.add_argument('--no-video', action='store_true',
                    help='Disable MP4 preview generation')
    ap.add_argument('--synthetic', action='store_true',
                    help='Generate and use synthetic data, ignoring --h5')
    args = ap.parse_args()

    h5_filepath = Path(args.h5)
    if args.synthetic:
        print('Generating synthetic event data of a moving box...')
        x, y, p, t = generate_synthetic_events()
        h5_filepath = Path('synthetic.hdf5')
        with h5py.File(h5_filepath, 'w') as f:
            g = f.create_group('/CD/events')
            g.create_dataset('x', data=x, dtype=np.int16)
            g.create_dataset('y', data=y, dtype=np.int16)
            g.create_dataset('p', data=p, dtype=np.int8)
            g.create_dataset('t', data=t, dtype=np.int64)
        print(f'✔︎ Synthetic data saved to {h5_filepath}')

    simulate(h5_filepath,
             version=args.version,
             slice_us=args.slice_us,
             active_v=args.active_v,
             silent_v=args.silent_v,
             save_video=not args.no_video,
             polarity=args.polarity)


if __name__ == '__main__':
    main() 