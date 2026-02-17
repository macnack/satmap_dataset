from __future__ import annotations

from satmap_dataset.models import YearStatus


def mock_statuses_2015_2026() -> list[YearStatus]:
    has_features = {2015, 2017, 2018, 2019, 2021, 2023}
    zero_features = {2016, 2020, 2022, 2024, 2025}
    no_typename = {2026}

    statuses: list[YearStatus] = []
    for year in range(2015, 2027):
        if year in has_features:
            statuses.append(
                YearStatus(
                    year=year,
                    typename_exists=True,
                    feature_count=2,
                    status="has_features",
                    reason=None,
                )
            )
        elif year in zero_features:
            statuses.append(
                YearStatus(
                    year=year,
                    typename_exists=True,
                    feature_count=0,
                    status="zero_features",
                    reason="Typename exists but no features found for AOI.",
                )
            )
        elif year in no_typename:
            statuses.append(
                YearStatus(
                    year=year,
                    typename_exists=False,
                    feature_count=0,
                    status="no_typename",
                    reason=f"No WFS typename found for year {year}.",
                )
            )
    return statuses


def mock_tiles_by_year_2015_2026() -> dict[int, dict[str, str]]:
    shared = "M-34-75-D-a-3-4"
    included = {2015, 2017, 2018, 2019, 2021, 2023}
    out: dict[int, dict[str, str]] = {}
    for year in range(2015, 2027):
        if year in included:
            out[year] = {
                shared: f"https://opendata.geoportal.gov.pl/ortofotomapa/{year}/{year}_x_{shared}.tif",
                f"extra_{year}": f"https://opendata.geoportal.gov.pl/ortofotomapa/{year}/{year}_x_extra_{year}.tif",
            }
        else:
            out[year] = {}
    return out
