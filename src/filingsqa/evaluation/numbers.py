"""How financial numbers are written in filings. Used to check that ground truth is present in
the corpus (dataset QA) and, in Phase 4, that cited chunks contain the cited number."""


def format_usd(value: float) -> str:
    """Human-readable reference value, in millions like the filings' tables."""
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value) / 1e6:,.0f} million"


def format_value(value: float, unit: str) -> str:
    if unit == "USD/shares":
        return f"${value:,.2f}"
    if unit == "USD":
        return format_usd(value)
    return f"{value:,.0f} {unit}"


def number_variants(value: float, unit: str) -> set[str]:
    """Strings under which the value can appear in filing text/tables (without '$')."""
    variants: set[str] = set()
    magnitude = abs(value)
    if unit == "USD/shares":
        variants.add(f"{magnitude:.2f}")
    elif unit == "USD":
        for scale in (1e6, 1e3, 1.0):  # tables "in millions", "in thousands", or raw
            scaled = magnitude / scale
            if scaled >= 1 and abs(scaled - round(scaled)) < 1e-9:
                variants.add(f"{round(scaled):,}")
        if not variants:  # not an exact multiple of a thousand: rounded millions
            variants.add(f"{magnitude / 1e6:,.0f}")
    else:
        variants.add(f"{magnitude:,.0f}")
    return variants
