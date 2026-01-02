"""
test_nitrate_units.py

Unit tests for nitrate basis conversion and unit conversion functions.
Tests the mathematical correctness of NO₃ ⇔ NO₃–N conversions.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from units import (
    convert_mgL,
    convert_nitrate_basis_mgL,
    detect_nitrate_basis_from_column,
    NO3_TO_NO3N,
    NO3N_TO_NO3,
    CF_ugL_to_mgL,
)


class TestNitrateConversion:
    """Tests for nitrate basis conversion (NO₃ ⇔ NO₃–N)."""
    
    def test_no3_to_no3n_who_guideline(self):
        """Test WHO guideline: 50 mg/L NO₃ → ~11.29 mg/L NO₃–N."""
        result = convert_nitrate_basis_mgL(50.0, 'NO3', 'NO3-N')
        expected = 50.0 * (14.0 / 62.0)  # ~11.29
        assert abs(result - expected) < 0.01
        assert abs(result - 11.29) < 0.01
    
    def test_no3n_to_no3_who_guideline(self):
        """Test reverse WHO guideline: 10 mg/L NO₃–N → ~44.29 mg/L NO₃."""
        result = convert_nitrate_basis_mgL(10.0, 'NO3-N', 'NO3')
        expected = 10.0 * (62.0 / 14.0)  # ~44.29
        assert abs(result - expected) < 0.01
        assert abs(result - 44.29) < 0.01
    
    def test_round_trip_no3(self):
        """Test round-trip conversion preserves value: NO₃ → NO₃–N → NO₃."""
        original = 50.0
        intermediate = convert_nitrate_basis_mgL(original, 'NO3', 'NO3-N')
        result = convert_nitrate_basis_mgL(intermediate, 'NO3-N', 'NO3')
        assert abs(result - original) < 1e-10
    
    def test_round_trip_no3n(self):
        """Test round-trip conversion preserves value: NO₃–N → NO₃ → NO₃–N."""
        original = 10.0
        intermediate = convert_nitrate_basis_mgL(original, 'NO3-N', 'NO3')
        result = convert_nitrate_basis_mgL(intermediate, 'NO3', 'NO3-N')
        assert abs(result - original) < 1e-10
    
    def test_same_basis_no_change(self):
        """Test that same basis returns unchanged value."""
        assert convert_nitrate_basis_mgL(50.0, 'NO3', 'NO3') == 50.0
        assert convert_nitrate_basis_mgL(10.0, 'NO3-N', 'NO3-N') == 10.0
    
    def test_basis_normalization_variants(self):
        """Test various input format normalizations."""
        # All these should be equivalent to NO3 -> NO3-N
        result1 = convert_nitrate_basis_mgL(50.0, 'NO3', 'NO3-N')
        result2 = convert_nitrate_basis_mgL(50.0, 'nitrate', 'NO3_N')
        result3 = convert_nitrate_basis_mgL(50.0, 'NO3', 'nitrate_N')
        
        assert abs(result1 - result2) < 1e-10
        assert abs(result1 - result3) < 1e-10
    
    def test_zero_value(self):
        """Test that zero converts to zero."""
        assert convert_nitrate_basis_mgL(0.0, 'NO3', 'NO3-N') == 0.0
        assert convert_nitrate_basis_mgL(0.0, 'NO3-N', 'NO3') == 0.0
    
    def test_conversion_constants(self):
        """Verify conversion constants are correct."""
        assert abs(NO3_TO_NO3N - 14.0/62.0) < 1e-10
        assert abs(NO3N_TO_NO3 - 62.0/14.0) < 1e-10
        assert abs(NO3_TO_NO3N * NO3N_TO_NO3 - 1.0) < 1e-10


class TestBasisDetection:
    """Tests for autodetecting nitrate basis from column names."""
    
    def test_detect_no3n_explicit(self):
        """Test detection of explicit NO₃–N columns."""
        assert detect_nitrate_basis_from_column('NO3_N') == 'NO3-N'
        assert detect_nitrate_basis_from_column('NO3-N') == 'NO3-N'
        assert detect_nitrate_basis_from_column('nitrate_N') == 'NO3-N'
        assert detect_nitrate_basis_from_column('Nitrate_N') == 'NO3-N'
    
    def test_detect_no3_default(self):
        """Test default detection of NO₃ columns."""
        assert detect_nitrate_basis_from_column('NO3') == 'NO3'
        assert detect_nitrate_basis_from_column('nitrate') == 'NO3'
        assert detect_nitrate_basis_from_column('Nitrate') == 'NO3'
    
    def test_detect_with_suffix(self):
        """Test detection handles _N suffix correctly."""
        assert detect_nitrate_basis_from_column('NO3_mgL') == 'NO3'
        # But if _N is in the name, it should detect as NO3-N
        assert detect_nitrate_basis_from_column('NO3_N_mgL') == 'NO3-N'


class TestConcentrationConversion:
    """Tests for general concentration unit conversion."""
    
    def test_ug_to_mg(self):
        """Test µg/L to mg/L conversion."""
        result = convert_mgL(1000.0, 'µg/L', 'mg/L')
        assert result == 1.0
    
    def test_mg_to_ug(self):
        """Test mg/L to µg/L conversion."""
        result = convert_mgL(1.0, 'mg/L', 'µg/L')
        assert result == 1000.0
    
    def test_same_unit_no_change(self):
        """Test same unit returns unchanged value."""
        assert convert_mgL(50.0, 'mg/L', 'mg/L') == 50.0
        assert convert_mgL(50.0, 'µg/L', 'µg/L') == 50.0
    
    def test_ug_variant_spelling(self):
        """Test 'ug/L' variant is handled correctly."""
        result = convert_mgL(1000.0, 'ug/L', 'mg/L')
        assert result == 1.0
    
    def test_cf_constant(self):
        """Verify conversion factor constant."""
        assert CF_ugL_to_mgL == 0.001


class TestMathematicalInvariants:
    """Tests for mathematical invariants required by blueprint."""
    
    def test_invariant_no3_conversion_exact(self):
        """Test invariant: C_NO3 × (14/62) == C_NO3N exactly."""
        c_no3 = 50.0
        c_no3n = convert_nitrate_basis_mgL(c_no3, 'NO3', 'NO3-N')
        expected = c_no3 * (14.0 / 62.0)
        assert abs(c_no3n - expected) < 1e-14  # Machine precision
    
    def test_invariant_no3n_conversion_exact(self):
        """Test invariant: C_NO3N × (62/14) == C_NO3 exactly."""
        c_no3n = 10.0
        c_no3 = convert_nitrate_basis_mgL(c_no3n, 'NO3-N', 'NO3')
        expected = c_no3n * (62.0 / 14.0)
        assert abs(c_no3 - expected) < 1e-14  # Machine precision


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
