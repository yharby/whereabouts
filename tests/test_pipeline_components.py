from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from duckdb import DuckDBPyConnection

from whereabouts.QueryPipeline import QueryPipeline
from whereabouts.matching_queries.common import register_functions
from whereabouts.matching_queries.standard_with_neighbours import (
    clean_addresses,
    create_address_numerics,
    create_address_alpha_tokens,
    create_input_phrases,
    first_matching_step,
    unnest_match_candidates,
    extract_match_candidate_details,
    filter_to_top50_candidates,
    neighbouring_suburb_match,
    compute_similarity,
    rank_by_similarity,
    select_current,
    rejoin_all_inputs
)            


@pytest.fixture(scope="module")
def pipeline_connection() -> DuckDBPyConnection:
    """Create one shared DuckDB connection used by all tests in this module."""
    con = duckdb.connect('whereabouts/models/db_test.db')
    # Keep tests independent of optional extension availability.
    con.execute("CREATE OR REPLACE MACRO unaccent(text_value) AS text_value;")
    yield con
    con.close()


@pytest.fixture()
def con(pipeline_connection: DuckDBPyConnection) -> DuckDBPyConnection:
    """Create a temporary input table with example addresses for each test."""
    pipeline_connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE input_addresses AS
        SELECT *
        FROM (
            VALUES
                (1, '115 sydney rd brunswick vic 3056'),
                (2, '504 Sydney Rd, Brunswick'),
                (3, '62 dawson st brunswick east')
        ) AS t(address_id, address);
        """
    )
    yield pipeline_connection
    pipeline_connection.execute("DROP TABLE IF EXISTS input_addresses;")

@pytest.mark.order(1)
def test_template_non_empty() -> None:
    assert clean_addresses.query_template is not None
    assert create_address_numerics.query_template is not None
    assert create_address_alpha_tokens.query_template is not None
    assert create_input_phrases.query_template is not None
    assert first_matching_step.query_template is not None
    assert unnest_match_candidates.query_template is not None
    assert extract_match_candidate_details.query_template is not None
    assert filter_to_top50_candidates.query_template is not None
    assert neighbouring_suburb_match.query_template is not None
    assert compute_similarity.query_template is not None
    assert rank_by_similarity.query_template is not None
    assert select_current.query_template is not None
    assert rejoin_all_inputs.query_template is not None


@pytest.mark.order(2)
def test_clean_addresses(con: DuckDBPyConnection) -> None:
    pipeline = QueryPipeline(con=con, steps=[clean_addresses])
    result = pipeline.execute()
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 3
    assert result.shape[1] == 2
    assert set(result.columns) == {"address_id", "address"}
    assert result.to_dict() == {'address_id': {0: 1, 1: 2, 2: 3}, 'address': {0: '115 SYDNEY RD BRUNSWICK VIC 3056', 1: '504 SYDNEY RD BRUNSWICK', 2: '62 DAWSON ST BRUNSWICK EAST'}}

@pytest.mark.order(3)
def test_create_address_numerics(con: DuckDBPyConnection) -> None:
    pipeline = QueryPipeline(con=con, steps=[clean_addresses, create_address_numerics])
    result = pipeline.execute()
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 3
    assert result.shape[1] == 3
    assert set(result.columns) == {"address_id", "address", 'numeric_tokens'}
    assert result['address_id'].tolist() == [1, 2, 3]
    assert result['address'].tolist() == [
        '115 SYDNEY RD BRUNSWICK VIC 3056',
        '504 SYDNEY RD BRUNSWICK',
        '62 DAWSON ST BRUNSWICK EAST',
    ]
    assert [tokens.tolist() for tokens in result['numeric_tokens']] == [
        ['115', '3056'],
        ['504'],
        ['62'],
    ]

@pytest.mark.order(4)
def test_create_address_alpha_tokens(con: DuckDBPyConnection) -> None:
    pipeline = QueryPipeline(con=con, steps=[clean_addresses, create_address_numerics, create_address_alpha_tokens])
    result = pipeline.execute()
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 3
    assert result.shape[1] == 4
    assert set(result.columns) == {"address_id", "address", 'numeric_tokens', 'alpha_tokens'}
    assert result['address_id'].tolist() == [1, 2, 3]
    assert [tokens.tolist() for tokens in result['numeric_tokens']] == [
        ['115', '3056'],
        ['504'],
        ['62'],
    ]
    assert [tokens.tolist() for tokens in result['alpha_tokens']] == [
        ['SYDNEY', 'RD', 'BRUNSWICK', 'VIC'],
        ['SYDNEY', 'RD', 'BRUNSWICK'],
        ['DAWSON', 'ST', 'BRUNSWICK', 'EAST'],
    ]

@pytest.mark.order(5)
def test_create_input_phrases(con: DuckDBPyConnection) -> None:
    pipeline = QueryPipeline(con=con, 
                             steps=[clean_addresses, 
                                    create_address_numerics, 
                                    create_address_alpha_tokens, 
                                    create_input_phrases])
    result = pipeline.execute()
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 15
    assert result.shape[1] == 2
    assert set(result.columns) == {"address_id", "tokenphrase"}
    assert result.to_dict() == {'address_id': {14: 1, 11: 1, 3: 1, 6: 1, 9: 1, 0: 1, 12: 2, 7: 2, 1: 2, 4: 2, 10: 3, 13: 3, 2: 3, 8: 3, 5: 3}, 
                                'tokenphrase': {14: '115 RD', 11: '115 SYDNEY', 3: 'BRUNSWICK VIC', 6: 'RD BRUNSWICK', 9: 'SYDNEY RD', 0: 'VIC 3056', 12: '504 RD', 7: '504 SYDNEY', 1: 'RD BRUNSWICK', 4: 'SYDNEY RD', 10: '62 DAWSON', 13: '62 ST', 2: 'BRUNSWICK EAST', 8: 'DAWSON ST', 5: 'ST BRUNSWICK'}}
    
@pytest.mark.order(6)
def test_first_matching_step(con: DuckDBPyConnection) -> None:
    pipeline = QueryPipeline(con=con, 
                             steps=[clean_addresses, 
                                    create_address_numerics, 
                                    create_address_alpha_tokens, 
                                    create_input_phrases, 
                                    first_matching_step])
    result = pipeline.execute()
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 15
    assert result.shape[1] == 2
    assert set(result.columns) == {"address_id", "address_ids2"}
    assert result['address_id'].to_dict() == {0: 3, 1: 3, 2: 1, 3: 2, 4: 1, 5: 2, 6: 1, 7: 3, 8: 1, 9: 3, 10: 1, 11: 1, 12: 2, 13: 2, 14: 3}
    assert result['address_ids2'].to_dict()[0] == 12721438

@pytest.mark.order(7)
def test_first_matching_step(con: DuckDBPyConnection) -> None:
    pipeline = QueryPipeline(con=con, 
                             steps=[clean_addresses, 
                                    create_address_numerics, 
                                    create_address_alpha_tokens, 
                                    create_input_phrases, 
                                    first_matching_step, 
                                    unnest_match_candidates])
    result = pipeline.execute()
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 112
    assert result.shape[1] == 2
    assert set(result.columns) == {"address_id1", "address_id2"}
    assert set(result['address_id1']) == {3, 1}
    assert len(set(result['address_id2'])) == 112

@pytest.mark.order(8)
def test_first_matching_step(con: DuckDBPyConnection) -> None:
    pipeline = QueryPipeline(con=con, 
                             steps=[clean_addresses, 
                                    create_address_numerics, 
                                    create_address_alpha_tokens, 
                                    create_input_phrases, 
                                    first_matching_step])
    result = pipeline.execute()
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 112
    assert result.shape[1] == 2
    assert set(result.columns) == {"address_id", "address_id2"}
    assert set(result['address_id1']) == {3, 1}
    assert len(set(result['address_id2'])) == 112