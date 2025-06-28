"""
Tests for the aggregate_data_by_month function.

This function aggregates biweekly simulation data into monthly totals,
performing both spatial (across group_ids) and temporal (across ticks) aggregation.
"""

import pytest
import polars as pl
import numpy as np
from datetime import datetime
import sys
import os

# Add the project source to the path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))
import project


class TestAggregateDataByMonth:
    """Test cases for aggregate_data_by_month function."""

    def test_basic_aggregation(self):
        """Test basic monthly aggregation with simple data."""
        # Create test data: 4 biweekly ticks (2 months) for 2 regions
        data = pl.DataFrame({
            "tick": [0, 1, 2, 3, 0, 1, 2, 3],
            "group_id": ["nigeria:kano:patch1", "nigeria:kano:patch1", 
                        "nigeria:kano:patch1", "nigeria:kano:patch1",
                        "nigeria:kano:patch2", "nigeria:kano:patch2",
                        "nigeria:kano:patch2", "nigeria:kano:patch2"],
            "cases": [10, 20, 30, 40, 5, 15, 25, 35]
        })
        
        result = project.aggregate_data_by_month(
            data=data,
            group_id="nigeria:kano",
            start_date="2000-01",
            target_times=["2000-01", "2000-02"], 
            tick_length=14,
            column="cases"
        )
        
        # Should aggregate spatially and temporally
        # Month 1 (ticks 0,1): (10+20) + (5+15) = 50
        # Month 2 (ticks 2,3): (30+40) + (25+35) = 130
        assert len(result) == 2
        assert result[0] == 50.0
        assert result[1] == 130.0

    def test_spatial_filtering(self):
        """Test that spatial filtering works correctly with group_id prefix matching."""
        data = pl.DataFrame({
            "tick": [0, 0, 0],
            "group_id": ["nigeria:kano:patch1", "nigeria:kaduna:patch1", "nigeria:kano:patch2"],
            "cases": [10, 100, 20]  # Only kano patches should be included
        })
        
        result = project.aggregate_data_by_month(
            data=data,
            group_id="nigeria:kano",
            start_date="2000-01",
            target_times=["2000-01"],
            tick_length=14,
            column="cases"
        )
        
        # Should only include nigeria:kano patches (10 + 20 = 30), not kaduna (100)
        assert len(result) == 1
        assert result[0] == 30.0

    def test_temporal_aggregation(self):
        """Test temporal aggregation across multiple ticks per month."""
        # 6 ticks, 14 days each = 84 days ≈ 3 months
        data = pl.DataFrame({
            "tick": [0, 1, 2, 3, 4, 5],
            "group_id": ["nigeria:test"] * 6,
            "cases": [10, 20, 30, 40, 50, 60]
        })
        
        result = project.aggregate_data_by_month(
            data=data,
            group_id="nigeria:test",
            start_date="2000-01",
            target_times=["2000-01", "2000-02", "2000-03"],
            tick_length=14,
            column="cases"
        )
        
        # Month 1: ticks 0,1 (days 0-27) = 10 + 20 = 30
        # Month 2: ticks 2,3 (days 28-55) = 30 + 40 = 70  
        # Month 3: ticks 4,5 (days 56-83) = 50 + 60 = 110
        assert len(result) == 3
        assert result[0] == 30.0
        assert result[1] == 70.0
        assert result[2] == 110.0

    def test_empty_data_handling(self):
        """Test handling of empty DataFrame."""
        data = pl.DataFrame({
            "tick": [],
            "group_id": [],
            "cases": []
        }, schema={"tick": pl.Int64, "group_id": pl.Utf8, "cases": pl.Float64})
        
        result = project.aggregate_data_by_month(
            data=data,
            group_id="nigeria:test",
            start_date="2000-01",
            target_times=["2000-01"],
            tick_length=14,
            column="cases"
        )
        
        assert result is None

    def test_no_matching_group_id(self):
        """Test handling when no data matches the group_id filter."""
        data = pl.DataFrame({
            "tick": [0, 1],
            "group_id": ["nigeria:kaduna:patch1", "nigeria:lagos:patch1"],
            "cases": [10, 20]
        })
        
        result = project.aggregate_data_by_month(
            data=data,
            group_id="nigeria:kano",  # No matching data
            start_date="2000-01",
            target_times=["2000-01"],
            tick_length=14,
            column="cases"
        )
        
        assert result is None

    def test_different_column_types(self):
        """Test aggregation with different column types (e.g., population counts)."""
        data = pl.DataFrame({
            "tick": [0, 1],
            "group_id": ["nigeria:test:patch1", "nigeria:test:patch1"],
            "cases": [10, 20],
            "count": [1000, 1100]
        })
        
        # Test with cases column
        result_cases = project.aggregate_data_by_month(
            data=data,
            group_id="nigeria:test",
            start_date="2000-01",
            target_times=["2000-01"],
            tick_length=14,
            column="cases"
        )
        
        # Test with count column  
        result_count = project.aggregate_data_by_month(
            data=data,
            group_id="nigeria:test",
            start_date="2000-01",
            target_times=["2000-01"],
            tick_length=14,
            column="count"
        )
        
        assert result_cases[0] == 30.0  # 10 + 20
        assert result_count[0] == 2100.0  # 1000 + 1100

    def test_multi_month_range(self):
        """Test aggregation over multiple months matching real usage patterns."""
        # Create data spanning several months
        ticks = list(range(0, 52))  # 52 biweekly ticks ≈ 2 years
        data = pl.DataFrame({
            "tick": ticks,
            "group_id": ["nigeria:kano:patch1"] * 26 + ["nigeria:kano:patch2"] * 26,
            "cases": [float(i + 1) for i in ticks]  # Cases increase over time
        })
        
        # Test 6 months of aggregation
        target_times = [f"2000-{i:02d}" for i in range(1, 7)]
        
        result = project.aggregate_data_by_month(
            data=data,
            group_id="nigeria:kano",
            start_date="2000-01",
            target_times=target_times,
            tick_length=14,
            column="cases"
        )
        
        assert len(result) == 6
        # Each result should be positive and increasing over time
        assert all(r > 0 for r in result)
        # Results should generally increase (more cases over time)
        assert result[5] > result[0]

    def test_edge_case_single_tick_per_month(self):
        """Test edge case where there's only one tick per month."""
        data = pl.DataFrame({
            "tick": [0, 2, 4],  # Sparse ticks
            "group_id": ["nigeria:test"] * 3,
            "cases": [100, 200, 300]
        })
        
        result = project.aggregate_data_by_month(
            data=data,
            group_id="nigeria:test",
            start_date="2000-01",
            target_times=["2000-01", "2000-02", "2000-03"],
            tick_length=14,
            column="cases"
        )
        
        # Each month should capture one tick
        assert len(result) == 3
        assert result[0] == 100.0  # tick 0
        assert result[1] == 200.0  # tick 2  
        assert result[2] == 300.0  # tick 4

    def test_return_type_and_format(self):
        """Test that function returns the correct type and format."""
        data = pl.DataFrame({
            "tick": [0, 1],
            "group_id": ["nigeria:test"] * 2,
            "cases": [10.5, 20.7]
        })
        
        result = project.aggregate_data_by_month(
            data=data,
            group_id="nigeria:test",
            start_date="2000-01",
            target_times=["2000-01"],
            tick_length=14,
            column="cases"
        )
        
        # Should return a list of floats
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], float)
        assert result[0] == 31.2  # 10.5 + 20.7


if __name__ == "__main__":
    # Run the tests
    test_instance = TestAggregateDataByMonth()
    
    # Run each test method
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    for method_name in test_methods:
        try:
            print(f"Running {method_name}...")
            method = getattr(test_instance, method_name)
            method()
            print(f"✓ {method_name} passed")
        except Exception as e:
            print(f"✗ {method_name} failed: {e}")
            raise
    
    print(f"\n✓ All {len(test_methods)} tests passed!")