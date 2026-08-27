import logging

import pandas as pd
import pytest

logging.getLogger("recordlinkage").setLevel(logging.ERROR)


@pytest.fixture
def master_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company": ["Apple Inc", "Microsoft Corporation", "Acme Corporation"],
            "country": ["US", "US", "UK"],
        }
    )


@pytest.fixture
def using_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company": ["Apple", "Microsoft Corp", "Acme Corp", "Banana Stand"],
            "country": ["US", "US", "UK", "US"],
        }
    )
