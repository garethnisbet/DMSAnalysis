# Example data

## `913232/` — icosahedral Al-Pd-Mn, energy scan

| | |
|---|---|
| Beamline | Diamond Light Source i16, visit mm29043-1, 5 Dec 2021 |
| Sample | iAlPdMn, not annealed (`crystal_name=iAlPdMn`) |
| Scan | `scan energy 6.295 6.303 0.0005 BeamOK pil2m 300` — 17 points, Pilatus 2M, 300 s each |
| Primary reflection | `h k l = 1.929 3.983 1.395` at `en = 6.299 keV`, azimuthal reference `0 1.5748 2.548` |
| Included | `913232.dat` (complete) and **one** frame, `913232-pilatus2M-files/00011.tif` — datapoint 10, the one `configs/example_913232.json` fits |

Only the frame the example reads is included: the full scan is 161 MB, against
a 3.4 MB frame here. The frame is the beamline's int32 image re-saved with
lossless deflate compression (9.9 MB → 3.4 MB); every pixel value, including
the negative module-gap markers, is unchanged, and `imageio` reads it exactly as
it reads the original. Stepping the slider to another datapoint therefore shows
a blank frame — the `.dat` is complete, so the energy, hkl and everything else
read from it are correct at every point.

The layout mirrors the beamline's (`<scan>.dat` beside
`<scan>-pilatus2M-files/%05d.tif`), so nothing in the apps special-cases it.
`configs/example_913232.json` is `configs/fit_fivefold_axis_AlPdMn_Not_Annealed_2M_2ROIS_internal_hkl.json`
with `scan.scanpath` changed to `../example_data/913232/`, which resolves
relative to the config file (`dat2config.resolve_scanpath`), and `zoomval` set
to 1: the sub-pixel ROI integrator interpolates, so the 2× zoom is no longer
needed.
