import argparse
import numpy as np
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib as mpl
    MATPLOT_OK = True
except Exception:
    plt = None
    animation = None
    mpl = None
    MATPLOT_OK = False
from pathlib import Path
import gzip
import json
from typing import Optional


def _load_meta(npz_path: Path) -> dict:
    meta_path = npz_path.with_suffix('.json.gz')
    if not meta_path.exists():
        return {}
    try:
        with gzip.open(meta_path, 'rt') as fp:
            return json.load(fp)
    except Exception:
        return {}


def _resistance_to_state_w(R: np.ndarray, ron: float, roff: float) -> np.ndarray:
    lam = float(np.log(roff / ron))
    eps = 1e-30
    return 1.0 - (np.log(np.maximum(R / ron, eps)) / lam)


def _save_keyframes(anim_data: np.ndarray,
                    npz_path: Path,
                    key_every: int,
                    key_dir: Optional[Path],
                    fmt: str,
                    cmap_name: str,
                    vmin: float,
                    vmax: float,
                    fps: float,
                    label: str):
    if key_every is None or key_every <= 0:
        return

    out_dir = key_dir or npz_path.parent / f"{npz_path.stem}_keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_npz": str(npz_path),
        "key_every": int(key_every),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "fps": float(fps),
        "frames": []
    }

    if MATPLOT_OK:
        cmap = plt.get_cmap(cmap_name)
        for idx in range(0, anim_data.shape[0], key_every):
            frame = anim_data[idx]
            out_path = out_dir / f"frame_{idx:05d}.{fmt}"
            plt.imsave(out_path, frame, cmap=cmap, vmin=vmin, vmax=vmax)
            manifest["frames"].append({"index": int(idx), "time_s": float(idx / fps), "path": out_path.name})
    else:
        import cv2
        cv2_cmap = cv2.COLORMAP_JET
        for idx in range(0, anim_data.shape[0], key_every):
            frame = anim_data[idx]
            frame_norm = (frame - vmin) / (vmax - vmin + 1e-12)
            frame_u8 = np.clip(frame_norm * 255.0, 0, 255).astype(np.uint8)
            frame_color = cv2.applyColorMap(frame_u8, cv2_cmap)
            out_path = out_dir / f"frame_{idx:05d}.png"
            cv2.imwrite(str(out_path), frame_color)
            manifest["frames"].append({"index": int(idx), "time_s": float(idx / fps), "path": out_path.name})

    with open(out_dir / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"✔︎ Saved {len(manifest['frames'])} keyframes to {out_dir}")


def _save_colorbar_image(npz_path: Path,
                         cmap_name: str,
                         vmin: float,
                         vmax: float,
                         label: str,
                         out_path: Optional[Path] = None,
                         orientation: str = 'horizontal'):
    """
    Save a standalone colorbar image using the same vmin/vmax and colormap as the animation.
    """
    out_path = out_path or npz_path.with_suffix('.colorbar.png')

    if MATPLOT_OK:
        # Standalone colorbar figure
        fig = plt.figure(figsize=(6, 1.0), dpi=200)
        ax = fig.add_axes([0.05, 0.25, 0.9, 0.5])  # [left, bottom, width, height]
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_name)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax, orientation=orientation)
        cbar.set_label(label)
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✔︎ Saved colorbar to {out_path}")
    else:
        # OpenCV fallback: create a gradient bar with numeric end labels
        import cv2
        width, height = 600, 60
        gradient = np.tile(np.linspace(0, 1, width, dtype=np.float32), (height, 1))
        gradient_u8 = np.clip(gradient * 255.0, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(gradient_u8, cv2.COLORMAP_JET)
        # Add simple text labels for vmin/vmax
        cv2.putText(color, f"{vmin:.3g}", (5, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        text = f"{label}  →  {vmax:.3g}"
        cv2.putText(color, text, (width - 5 - 8*len(text), height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_path), color)
        print(f"✔︎ Saved colorbar (OpenCV) to {out_path}")


def visualize_npz(npz_path: Path,
                  mode: str = 'abs',
                  use_log: bool = False,
                  out_mp4: bool = False,
                  fps: Optional[float] = None,
                  value: str = 'resistance',
                  key_every: int = 0,
                  key_dir: Optional[Path] = None,
                  key_format: str = 'png',
                  save_colorbar: bool = True,
                  colorbar_path: Optional[Path] = None):
    """
    Loads a simulation result .npz file and visualizes its contents.
    - Plots the final state `w_final`.
    - Creates an animation in resistance space R(t) or state space w(t).
    - Optionally saves keyframes every N frames into a subfolder.
    - Saves a standalone colorbar image using the same normalization and colormap.
    """
    try:
        data = np.load(npz_path)
    except FileNotFoundError:
        print(f"Error: File not found at {npz_path}")
        return

    if 'w_final' not in data or 'resistances' not in data:
        print(f"Error: NPZ file {npz_path} is missing 'w_final' or 'resistances' key.")
        return

    w_final = data['w_final']
    resistances = data['resistances']

    meta = _load_meta(npz_path)
    if fps is None:
        fps = float(meta.get('fps', 30.0)) if meta else 30.0

    # 1. Final w plot
    fig_w, ax_w = (plt.subplots() if MATPLOT_OK else (None, None))
    if MATPLOT_OK:
        im_w = ax_w.imshow(w_final, cmap='viridis')
        ax_w.set_title(f'Final State (w_final)\\n{npz_path.name}')
        (fig_w.colorbar(im_w, ax=ax_w, label='State variable w') if fig_w else None)
        w_final_path = npz_path.with_suffix('.w_final.png')
        plt.savefig(w_final_path, bbox_inches='tight')
        print(f'✔︎ Saved final state plot to {w_final_path}')
        plt.close(fig_w)
    else:
        import cv2
        w_norm = cv2.normalize(w_final, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        w_final_path = npz_path.with_suffix('.w_final.png')
        cv2.imwrite(str(w_final_path), w_norm)
        print(f'✔︎ Saved final state plot to {w_final_path}')

    # 2. Build animation data
    if resistances.ndim != 3 or resistances.shape[0] == 0:
        print("Skipping animation: 'resistances' array is not suitable for animation.")
        return

    if value == 'resistance':
        base = resistances
        base_label = 'Resistance (Ω)'
        def to_delta(B, B0, eps):
            return B0 - B
        def to_rel(B, B0, eps):
            return (B0 - B) / (B0 + eps)
    elif value == 'state':
        params = (meta.get('params') if meta else None) or {}
        ron = float(params.get('Ron', 1.0))
        roff = float(params.get('Roff', 2.0))
        base = _resistance_to_state_w(resistances, ron=ron, roff=roff)
        base_label = 'State w (0–1)'
        def to_delta(B, B0, eps):
            return B - B0
        def to_rel(B, B0, eps):
            return (B - B0) / (np.abs(B0) + eps)
    else:
        print("Unknown value. Use 'resistance' or 'state'.")
        return

    B0 = base[0]
    eps = 1e-9
    if mode == 'abs':
        anim_data = base
        label = base_label
    elif mode == 'delta':
        anim_data = to_delta(base, B0, eps)
        label = f'Δ{base_label}'
    elif mode == 'rel':
        anim_data = to_rel(base, B0, eps)
        label = f'Relative change Δ/{"R0" if value=="resistance" else "w0"}'
    else:
        print(f"Unknown mode: {mode}. Use one of: abs, delta, rel.")
        return

    if use_log:
        anim_data = np.log10(np.maximum(anim_data, eps))
        label = f'log10({label})'

    vmin = float(np.nanmin(anim_data))
    vmax = float(np.nanmax(anim_data))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12

    # Save keyframes first
    _save_keyframes(anim_data=anim_data,
                    npz_path=npz_path,
                    key_every=key_every,
                    key_dir=Path(key_dir) if key_dir else None,
                    fmt=key_format,
                    cmap_name='inferno',
                    vmin=vmin,
                    vmax=vmax,
                    fps=fps,
                    label=label)

    # Save a standalone colorbar using the SAME normalization and cmap as the animation
    if save_colorbar:
        cb_path = Path(colorbar_path) if colorbar_path else npz_path.with_suffix('.colorbar.png')
        _save_colorbar_image(npz_path=npz_path,
                             cmap_name='inferno',
                             vmin=vmin,
                             vmax=vmax,
                             label=label,
                             out_path=cb_path,
                             orientation='horizontal')

    # Animation
    if MATPLOT_OK:
        fig_anim, ax_anim = plt.subplots()
        im_anim = ax_anim.imshow(anim_data[0], cmap='inferno', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax_anim.set_title(f'{"State" if value=="state" else "Resistance"} History ({mode})\\n{npz_path.name}')
        fig_anim.colorbar(im_anim, ax=ax_anim, label=label)

        def update(frame):
            im_anim.set_data(anim_data[frame])
            return [im_anim]

        anim_obj = animation.FuncAnimation(fig_anim, update, frames=len(anim_data), blit=True)

        if out_mp4:
            animation_path = npz_path.with_suffix('.animation.mp4')
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
                anim_obj.save(str(animation_path), writer=writer, dpi=120)
                print(f'✔︎ Saved animation to {animation_path} @ {fps:.2f} fps')
            except Exception as e:
                print(f"FFMpegWriter failed ({e}); falling back to GIF.")
                animation_path = npz_path.with_suffix('.animation.gif')
                anim_obj.save(str(animation_path), writer='pillow', fps=int(round(fps)), dpi=120)
                print(f'✔︎ Saved animation to {animation_path} @ {fps:.2f} fps')
        else:
            animation_path = npz_path.with_suffix('.animation.gif')
            anim_obj.save(str(animation_path), writer='pillow', fps=int(round(fps)), dpi=120)
            print(f'✔︎ Saved animation to {animation_path} @ {fps:.2f} fps')

        plt.close(fig_anim)
    else:
        import cv2
        H, W = anim_data.shape[1:]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = npz_path.with_suffix('.animation.mp4')
        vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H), isColor=True)
        for idx in range(anim_data.shape[0]):
            frame = anim_data[idx]
            frame_norm = (frame - vmin) / (vmax - vmin)
            frame_u8 = np.clip(frame_norm * 255.0, 0, 255).astype(np.uint8)
            frame_color = cv2.applyColorMap(frame_u8, cv2.COLORMAP_JET)
            vw.write(frame_color)
        vw.release()
        print(f'✔︎ Saved animation to {out_path} @ {fps:.2f} fps')


def main():
    ap = argparse.ArgumentParser(description="Visualize simulation results from .npz file, with keyframes and colorbar export.")
    ap.add_argument('npz_file', type=str, help="Path to the .npz file to visualize.")
    ap.add_argument('--mode', choices=['abs', 'delta', 'rel'], default='abs', help="Quantity to animate: abs, delta, or rel")
    ap.add_argument('--log', action='store_true', help="Apply log10 scaling to the animated quantity")
    ap.add_argument('--mp4', action='store_true', help="Save MP4 (requires ffmpeg); default is GIF")
    ap.add_argument('--fps', type=float, default=None, help="Override FPS for animation; default uses metadata JSON if available, else 30")
    ap.add_argument('--value', choices=['resistance', 'state'], default='resistance', help="Animate in resistance space R or state space w")
    ap.add_argument('--key-every', type=int, default=0, help="Save every Nth frame as a still (0=disabled)")
    ap.add_argument('--key-dir', type=str, default=None, help="Directory for keyframes (default: <npz_stem>_keyframes)")
    ap.add_argument('--key-format', choices=['png','jpg'], default='png', help="Image format for keyframes (matplotlib path)")
    ap.add_argument('--save-colorbar', dest='save_colorbar', action='store_true', default=True, help="Save a standalone colorbar image (default: on)")
    ap.add_argument('--no-save-colorbar', dest='save_colorbar', action='store_false', help="Disable saving the standalone colorbar image")
    ap.add_argument('--colorbar-path', type=str, default=None, help="Path for the colorbar image (default: <npz>.colorbar.png)")
    args = ap.parse_args()

    key_dir = Path(args.key_dir) if args.key_dir else None
    colorbar_path = Path(args.colorbar_path) if args.colorbar_path else None

    if MATPLOT_OK:
        visualize_npz(Path(args.npz_file),
                      mode=args.mode,
                      use_log=args.log,
                      out_mp4=args.mp4,
                      fps=args.fps,
                      value=args.value,
                      key_every=args.key_every,
                      key_dir=key_dir,
                      key_format=args.key_format,
                      save_colorbar=args.save_colorbar,
                      colorbar_path=colorbar_path)
    else:
        print('matplotlib not available. Using OpenCV fallback (MP4 only, JET colormap).')
        visualize_npz(Path(args.npz_file),
                      mode=args.mode,
                      use_log=args.log,
                      out_mp4=True,
                      fps=args.fps,
                      value=args.value,
                      key_every=args.key_every,
                      key_dir=key_dir,
                      key_format=args.key_format,
                      save_colorbar=args.save_colorbar,
                      colorbar_path=colorbar_path)


if __name__ == '__main__':
    main()
