#!/usr/bin/env python3
"""
Test pkdpipe campaign management functionality.

This module contains comprehensive tests for the campaign orchestration system,
including configuration validation, simulation variant management, dependency
resolution, and CLI interface testing.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from pkdpipe.campaign import (
    Campaign, 
    CampaignConfig, 
    SimulationVariant, 
    CampaignStatus, 
    SimulationStatus,
    CampaignError
)
from pkdpipe.config import COSMOLOGY_PRESETS, SIMULATION_PRESETS


class TestSimulationVariant:
    """Test the SimulationVariant dataclass."""
    
    def test_valid_variant_creation(self):
        """Test creating a valid simulation variant."""
        variant = SimulationVariant(
            name="test-variant",
            cosmology="lcdm",
            resolution="S0-validation",
            priority=75,
            dependencies=["other-variant"],
            custom_params={"nGrid": 1000}
        )
        
        assert variant.name == "test-variant"
        assert variant.cosmology == "lcdm"
        assert variant.resolution == "S0-validation"
        assert variant.priority == 75
        assert "other-variant" in variant.dependencies
        assert variant.custom_params["nGrid"] == 1000
    
    def test_invalid_cosmology_raises_error(self):
        """Test that invalid cosmology raises ValueError."""
        with pytest.raises(ValueError, match="Unknown cosmology preset"):
            SimulationVariant(
                name="test-variant",
                cosmology="nonexistent-cosmology",
                resolution="S0-validation"
            )
    
    def test_invalid_resolution_raises_error(self):
        """Test that invalid resolution raises ValueError."""
        with pytest.raises(ValueError, match="Unknown simulation preset"):
            SimulationVariant(
                name="test-variant",
                cosmology="lcdm",
                resolution="nonexistent-resolution"
            )


class TestCampaignConfig:
    """Test the CampaignConfig class."""
    
    def create_test_variant(self, name="test-variant"):
        """Helper to create a test variant."""
        return SimulationVariant(
            name=name,
            cosmology="lcdm", 
            resolution="S0-validation"
        )
    
    def test_valid_config_creation(self):
        """Test creating a valid campaign configuration."""
        variants = [self.create_test_variant()]
        config = CampaignConfig(
            name="test-campaign",
            description="Test campaign",
            variants=variants,
            max_concurrent_jobs=5
        )
        
        assert config.name == "test-campaign"
        assert config.description == "Test campaign"
        assert len(config.variants) == 1
        assert config.max_concurrent_jobs == 5
    
    def test_empty_variants_raises_error(self):
        """Test that empty variants list raises ValueError."""
        with pytest.raises(ValueError, match="Campaign must have at least one simulation variant"):
            CampaignConfig(
                name="test-campaign",
                description="Test campaign",
                variants=[]
            )
    
    def test_duplicate_variant_names_raises_error(self):
        """Test that duplicate variant names raise ValueError."""
        variants = [
            self.create_test_variant("duplicate"),
            self.create_test_variant("duplicate")
        ]
        
        with pytest.raises(ValueError, match="Duplicate variant names found"):
            CampaignConfig(
                name="test-campaign",
                description="Test campaign",
                variants=variants
            )
    
    def test_from_yaml_valid_file(self):
        """Test loading campaign config from a valid YAML file."""
        yaml_content = {
            'name': 'test-yaml-campaign',
            'description': 'Test YAML campaign',
            'variants': [
                {
                    'name': 'test-variant',
                    'cosmology': 'lcdm',
                    'resolution': 'S0-validation',
                    'priority': 80,
                    'custom_params': {'nGrid': 500}
                }
            ],
            'max_concurrent_jobs': 3
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_file = f.name
        
        try:
            config = CampaignConfig.from_yaml(yaml_file)
            assert config.name == 'test-yaml-campaign'
            assert len(config.variants) == 1
            assert config.variants[0].name == 'test-variant'
            assert config.variants[0].priority == 80
            assert config.max_concurrent_jobs == 3
        finally:
            Path(yaml_file).unlink()
    
    def test_from_yaml_nonexistent_file(self):
        """Test that loading from nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CampaignConfig.from_yaml("nonexistent.yaml")


class TestCampaign:
    """Test the Campaign class."""
    
    def create_test_config(self):
        """Helper to create a test campaign configuration."""
        variants = [
            SimulationVariant(
                name="variant1",
                cosmology="lcdm",
                resolution="S0-validation",
                priority=80
            ),
            SimulationVariant(
                name="variant2", 
                cosmology="wcdm",
                resolution="S0-validation",
                priority=60,
                dependencies=["variant1"]
            )
        ]
        
        return CampaignConfig(
            name="test-campaign",
            description="Test campaign",
            variants=variants
        )
    
    @patch('pkdpipe.campaign.Simulation')
    def test_campaign_initialization(self, mock_simulation):
        """Test campaign initialization with mocked Simulation class."""
        config = self.create_test_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            campaign = Campaign(config)
            
            assert campaign.config == config
            assert campaign.status == CampaignStatus.PLANNING
            assert len(campaign.simulations) == 2
            assert "variant1" in campaign.simulations
            assert "variant2" in campaign.simulations
    
    @patch('pkdpipe.campaign.Simulation')
    def test_get_submittable_variants(self, mock_simulation):
        """Test getting variants ready for submission."""
        config = self.create_test_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            campaign = Campaign(config)
            
            # Initially, only variant1 should be submittable (no dependencies)
            submittable = campaign.get_submittable_variants()
            assert len(submittable) == 1
            assert submittable[0].name == "variant1"
            
            # After variant1 completes, variant2 should be submittable
            campaign.simulation_status["variant1"] = SimulationStatus.COMPLETED
            submittable = campaign.get_submittable_variants()
            assert len(submittable) == 1
            assert submittable[0].name == "variant2"
    
    @patch('pkdpipe.campaign.Simulation')
    def test_submit_variant_dry_run(self, mock_simulation):
        """Test dry run submission of a variant."""
        config = self.create_test_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            campaign = Campaign(config)
            
            # Test dry run
            success = campaign.submit_variant("variant1", dry_run=True)
            assert success
            # Status should not change in dry run
            assert campaign.simulation_status["variant1"] == SimulationStatus.NOT_SUBMITTED
    
    @patch('pkdpipe.campaign.Simulation')
    def test_submit_variant_real(self, mock_simulation):
        """Test real submission of a variant."""
        config = self.create_test_config()
        mock_sim_instance = MagicMock()
        mock_sim_instance.create.return_value = "12345678"  # Mock job ID
        mock_simulation.return_value = mock_sim_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            campaign = Campaign(config)
            
            # Test real submission
            success = campaign.submit_variant("variant1", dry_run=False)
            assert success
            mock_sim_instance.create.assert_called_once()
            assert campaign.simulation_status["variant1"] == SimulationStatus.QUEUED
            assert campaign.job_ids["variant1"] == "12345678"  # Check job ID is stored
    
    @patch('pkdpipe.campaign.Simulation')
    def test_submit_nonexistent_variant(self, mock_simulation):
        """Test submitting a nonexistent variant raises ValueError."""
        config = self.create_test_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            campaign = Campaign(config)
            
            with pytest.raises(ValueError, match="Unknown variant"):
                campaign.submit_variant("nonexistent-variant")
    
    @patch('pkdpipe.campaign.Simulation')
    def test_state_persistence(self, mock_simulation):
        """Test that campaign state is saved and loaded correctly."""
        config = self.create_test_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            
            # Create campaign and modify state
            campaign = Campaign(config)
            campaign.simulation_status["variant1"] = SimulationStatus.COMPLETED
            campaign.job_ids["variant1"] = "12345"
            campaign._save_state()
            
            # Create new campaign instance and check state is loaded
            new_campaign = Campaign(config)
            assert new_campaign.simulation_status["variant1"] == SimulationStatus.COMPLETED
            assert new_campaign.job_ids["variant1"] == "12345"
    
    @patch('pkdpipe.campaign.Simulation')
    def test_generate_report(self, mock_simulation):
        """Test generating campaign status report."""
        config = self.create_test_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            campaign = Campaign(config)
            
            # Set some status
            campaign.simulation_status["variant1"] = SimulationStatus.COMPLETED
            campaign.simulation_status["variant2"] = SimulationStatus.RUNNING

            report = campaign.generate_report()
            assert "test-campaign" in report
            assert "Completed:       1" in report
            assert "Running:         1" in report
            assert "variant1" in report
            assert "variant2" in report
    
    @patch('pkdpipe.campaign.Simulation')
    def test_list_variants_with_filter(self, mock_simulation):
        """Test listing variants with status filter."""
        config = self.create_test_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            campaign = Campaign(config)
            
            # Set different statuses
            campaign.simulation_status["variant1"] = SimulationStatus.COMPLETED
            campaign.simulation_status["variant2"] = SimulationStatus.RUNNING
            
            # Test filtering
            completed = campaign.list_variants(status_filter=SimulationStatus.COMPLETED)
            assert len(completed) == 1
            assert completed[0].name == "variant1"
            
            running = campaign.list_variants(status_filter=SimulationStatus.RUNNING)
            assert len(running) == 1
            assert running[0].name == "variant2"


class TestCampaignCLI:
    """Test the campaign CLI functionality."""
    
    def create_test_yaml_file(self, temp_dir):
        """Helper to create a test YAML file."""
        yaml_content = {
            'name': 'cli-test-campaign',
            'description': 'CLI test campaign',
            'variants': [
                {
                    'name': 'cli-test-variant',
                    'cosmology': 'lcdm',
                    'resolution': 'S0-validation',
                    'priority': 80
                }
            ]
        }
        
        yaml_file = Path(temp_dir) / "test-campaign.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        
        return str(yaml_file)
    
    @patch('pkdpipe.campaign_cli.Campaign')
    def test_create_campaign_cli(self, mock_campaign):
        """Test campaign creation via CLI."""
        from pkdpipe.campaign_cli import create_campaign
        
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = self.create_test_yaml_file(temp_dir)
            
            # Mock the args object
            args = MagicMock()
            args.config = yaml_file
            args.validate_only = False
            
            # Mock campaign instance
            mock_campaign_instance = MagicMock()
            mock_campaign_instance.config.name = "cli-test-campaign"
            mock_campaign_instance.config.variants = [MagicMock()]
            mock_campaign_instance.output_dir = Path(temp_dir)
            mock_campaign_instance.generate_report.return_value = "Test report"
            mock_campaign.return_value = mock_campaign_instance
            
            result = create_campaign(args)
            assert result == 0
            mock_campaign.assert_called_once_with(yaml_file)
    
    def test_validate_only_cli(self):
        """Test validate-only mode in CLI."""
        from pkdpipe.campaign_cli import create_campaign
        
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = self.create_test_yaml_file(temp_dir)
            
            # Mock the args object
            args = MagicMock()
            args.config = yaml_file
            args.validate_only = True
            
            result = create_campaign(args)
            assert result == 0


class TestCosmologyPresets:
    """Test the cosmology presets for campaign."""
    
    def test_lcdm_preset_exists(self):
        """Test that lcdm preset is properly defined."""
        assert "lcdm" in COSMOLOGY_PRESETS
        lcdm = COSMOLOGY_PRESETS["lcdm"]
        
        # Check key parameters
        assert "h" in lcdm
        assert "ombh2" in lcdm  
        assert "omch2" in lcdm
        assert "As" in lcdm
        assert "ns" in lcdm
        assert lcdm["description"] == "Standard LCDM cosmology based on DESI-DR2-Planck-ACT"
    
    def test_wcdm_preset_exists(self):
        """Test that wcdm preset is properly defined."""
        assert "wcdm" in COSMOLOGY_PRESETS
        wcdm = COSMOLOGY_PRESETS["wcdm"]
        
        # Check wCDM-specific parameters
        assert "w0" in wcdm
        assert "wa" in wcdm
        assert wcdm["w0"] == -0.838
        assert wcdm["wa"] == -0.62
        assert "evolving dark energy" in wcdm["description"]
    
    def test_phicdm_preset_exists(self):
        """Test that phicdm preset is properly defined."""
        assert "phicdm" in COSMOLOGY_PRESETS
        phicdm = COSMOLOGY_PRESETS["phicdm"]
        
        # Check phiCDM-specific parameters
        assert "phi_model" in phicdm
        assert "phi_params" in phicdm
        assert phicdm["phi_model"] == "quintessence"
        assert isinstance(phicdm["phi_params"], dict)
        assert "scalar field dark energy" in phicdm["description"]


class TestSimulationPresets:
    """Test the simulation presets for summer campaign."""
    
    def test_S0_validation_preset_exists(self):
        """Test that S0-validation preset is properly defined."""
        assert "S0-validation" in SIMULATION_PRESETS
        validation = SIMULATION_PRESETS["S0-validation"]
        
        assert validation["dBoxSize"] == 1050
        assert validation["nGrid"] == 1400
        assert validation["nodes"] == 2
        assert validation["gpupern"] == 4
        assert validation["tlimit"] == "48:00:00"  # Fixed: was 12:00:00, now correctly 48:00:00
    
    def test_S0_production_preset_exists(self):
        """Test that S0-production preset is properly defined."""
        assert "S0-production" in SIMULATION_PRESETS
        production = SIMULATION_PRESETS["S0-production"]
        
        assert production["dBoxSize"] == 5250
        assert production["nGrid"] == 7000
        assert production["nodes"] == 250
        assert production["gpupern"] == 4
        assert production["tlimit"] == "48:00:00"
    
    def test_S0_scaling_preset_exists(self):
        """Test that S0-scaling preset is properly defined."""
        assert "S0-scaling" in SIMULATION_PRESETS
        scaling = SIMULATION_PRESETS["S0-scaling"]
        
        assert scaling["dBoxSize"] == 2100
        assert scaling["nGrid"] == 2800
        assert scaling["nodes"] == 16
        assert scaling["gpupern"] == 4
        assert scaling["tlimit"] == "48:00:00"
    
    def test_S0_test_preset_exists(self):
        """Test that S0-test preset is properly defined (default simulation)."""
        assert "S0-test" in SIMULATION_PRESETS
        s0_test = SIMULATION_PRESETS["S0-test"]
        
        assert s0_test["dBoxSize"] == 525
        assert s0_test["nGrid"] == 700
        assert s0_test["nodes"] == 1
        assert s0_test["tlimit"] == "12:00:00"  # Shorter time for testing


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
