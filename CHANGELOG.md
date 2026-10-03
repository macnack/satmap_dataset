# Changelog

All notable changes to this project are documented in this file.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- MIT `LICENSE`, contributor docs, architecture / data-licensing / security notes
- GitHub Actions CI (Python 3.10–3.12, libvips, pytest)
- Neutral default paths for raw/gmix roots (`~/sat_data_raw`, `~/sat_data`,
  overridable via `SATMAP_RAW_ROOT` / `SATMAP_GMIX_DEST`)

### Changed

- README and `pyproject.toml` no longer describe the project as a Phase-1 scaffold
- `requirements.txt` synced with `pyproject.toml` core dependencies

### Removed

- Dead `pipeline/mosaic.py` stub (CLI `mosaic` remains a `render` alias)
