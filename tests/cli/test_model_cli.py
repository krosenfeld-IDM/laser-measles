"""
Tests for the model CLI functionality converted from Click to Typer.
"""

import pytest
from typer.testing import CliRunner
from laser_measles.nigeria.model import app as nigeria_app
from laser_measles.washington.model import app as washington_app


class TestNigeriaCLI:
    """Test suite for Nigeria model CLI."""
    
    def test_nigeria_cli_help(self):
        """Test that the Nigeria CLI help command works with typer."""
        runner = CliRunner()
        result = runner.invoke(nigeria_app, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        
    def test_nigeria_cli_options(self):
        """Test that the Nigeria CLI accepts the expected options."""
        runner = CliRunner()
        result = runner.invoke(nigeria_app, ["--help"])
        assert result.exit_code == 0
        # Check for key options
        assert "--nticks" in result.stdout
        assert "--seed" in result.stdout
        assert "--verbose" in result.stdout
        assert "--no-viz" in result.stdout
        assert "--pdf" in result.stdout


class TestWashingtonCLI:
    """Test suite for Washington model CLI."""
    
    def test_washington_cli_help(self):
        """Test that the Washington CLI help command works with typer."""
        runner = CliRunner()
        result = runner.invoke(washington_app, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        
    def test_washington_cli_options(self):
        """Test that the Washington CLI accepts the expected options."""
        runner = CliRunner()
        result = runner.invoke(washington_app, ["--help"])
        assert result.exit_code == 0
        # Check for key options
        assert "--nticks" in result.stdout
        assert "--seed" in result.stdout
        assert "--verbose" in result.stdout
        assert "--no-viz" in result.stdout
        assert "--pdf" in result.stdout