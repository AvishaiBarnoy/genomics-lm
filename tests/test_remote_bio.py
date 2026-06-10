import os
import shutil
import pytest
from src.eval import remote_bio

def test_mock_blast_query():
    seq = "MKLVFLVLLFLGAVG"
    result = remote_bio.mock_blast_query(seq)
    assert result["sequence"] == seq
    assert len(result["hits"]) > 0
    assert result["hits"][0]["species"] == "Escherichia coli"
    
    # Try seq without Methionine (M)
    result_no_m = remote_bio.mock_blast_query("AAAAA")
    assert result_no_m["hits"][0]["species"] == "Bacillus subtilis"

def test_sqlite_caching(tmp_path):
    # Override CACHE_DB_PATH for isolated test
    test_db = tmp_path / "test_cache.db"
    original_db = remote_bio.CACHE_DB_PATH
    remote_bio.CACHE_DB_PATH = str(test_db)
    
    try:
        seq = "MKLVFLVLLFLGAVG"
        # Initially cached should be None
        assert remote_bio.get_cached_result(seq) is None
        
        # Save a result
        mock_result = {"sequence": seq, "hits": [{"species": "Test Bacteria"}], "source": "Test Source"}
        remote_bio.save_to_cache(seq, mock_result)
        
        # Retrieve it
        cached = remote_bio.get_cached_result(seq)
        assert cached is not None
        assert cached["hits"][0]["species"] == "Test Bacteria"
        assert cached["source"] == "Test Source"
    finally:
        # Restore CACHE_DB_PATH
        remote_bio.CACHE_DB_PATH = original_db

def test_query_remote_blast_fallback():
    seq = "MKLVFLVLLFLGAVG"
    # Ensure remote is disabled
    remote_bio.REMOTE_ENABLED = False
    
    # Query should execute mock fallback and cache it automatically
    res = remote_bio.query_remote_blast(seq)
    assert res is not None
    assert "hits" in res
    assert "source" in res
    assert "Mock" in res["source"]
