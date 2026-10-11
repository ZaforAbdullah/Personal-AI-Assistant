import pytest

from app.config import Settings, asyncpg_url


@pytest.mark.parametrize(
    "given",
    [
        "postgres://u:p@h:5432/db",
        "postgresql://u:p@h:5432/db",
        "postgresql+psycopg://u:p@h:5432/db",
        "postgresql+psycopg2://u:p@h:5432/db",
        "postgresql+asyncpg://u:p@h:5432/db",
        "  postgresql://u:p@h:5432/db?sslmode=require  ",
    ],
)
def test_connection_strings_are_normalised_for_asyncpg(given: str) -> None:
    assert asyncpg_url(given) == "postgresql+asyncpg://u:p@h:5432/db"


def test_other_query_parameters_survive_removing_sslmode() -> None:
    url = "postgres://u:p@h/db?sslmode=require&application_name=x"
    assert asyncpg_url(url) == "postgresql+asyncpg://u:p@h/db?application_name=x"


def test_supabase_url_is_normalised_and_required() -> None:
    s = Settings(_env_file=None, db_provider="supabase", supabase_db_url="postgres://u:p@h/db")
    assert s.db_url == "postgresql+asyncpg://u:p@h/db"
    with pytest.raises(ValueError, match="SUPABASE_DB_URL"):
        _ = Settings(_env_file=None, db_provider="supabase").db_url
