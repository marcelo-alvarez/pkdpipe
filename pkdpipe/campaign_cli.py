#!/usr/bin/env python3
"""
Campaign CLI interface for pkdpipe.

This module provides command-line tools for managing multi-simulation campaigns,
including creation, submission, monitoring, and reporting functionality.
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from .campaign import Campaign, CampaignConfig, SimulationStatus
from .config import PkdpipeConfigError


def create_campaign(args) -> int:
    """Create a new campaign from YAML configuration."""
    try:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            return 1
        
        if args.validate_only:
            # Just validate the configuration without creating
            try:
                config = CampaignConfig.from_yaml(str(config_path))
                print(f"✓ Campaign configuration '{config.name}' is valid")
                print(f"  Description: {config.description}")
                print(f"  Variants: {len(config.variants)}")
                for variant in config.variants:
                    print(f"    - {variant.name} ({variant.cosmology}, {variant.resolution})")
                return 0
            except Exception as e:
                print(f"✗ Configuration validation failed: {e}")
                return 1
        
        # Create the campaign
        campaign = Campaign(str(config_path))
        
        print(f"✓ Campaign '{campaign.config.name}' created successfully")
        print(f"  Output directory: {campaign.output_dir}")
        print(f"  Variants: {len(campaign.config.variants)}")
        
        # Print summary
        report = campaign.generate_report()
        print("\n" + report)
        
        return 0
        
    except Exception as e:
        print(f"Error creating campaign: {e}")
        return 1


def submit_campaign(args) -> int:
    """Submit campaign simulations."""
    try:
        input_path = Path(args.campaign_dir)
        
        # Handle both YAML config files and campaign directories
        if input_path.is_file() and input_path.suffix in ['.yaml', '.yml']:
            # User provided YAML config file - derive campaign directory
            config_file = input_path
            if not config_file.exists():
                print(f"Error: Configuration file not found: {config_file}")
                return 1
            
            # Load config to get campaign name and determine output directory
            try:
                config = CampaignConfig.from_yaml(str(config_file))
                if config.output_dir:
                    campaign_dir = Path(config.output_dir)
                else:
                    from .config import DEFAULT_RUN_DIR_BASE
                    campaign_dir = Path(DEFAULT_RUN_DIR_BASE) / f"campaign-{config.name}"
            except Exception as e:
                print(f"Error loading configuration: {e}")
                return 1
                
        elif input_path.is_dir():
            # User provided campaign directory
            campaign_dir = input_path
            
            # Find the original config file in the directory
            config_files = list(campaign_dir.glob("*.yaml")) + list(campaign_dir.glob("*.yml"))
            if not config_files:
                print(f"Error: No YAML configuration file found in {campaign_dir}")
                return 1
            config_file = config_files[0]  # Use the first one found
            
        else:
            print(f"Error: Path must be either a YAML config file or campaign directory: {input_path}")
            return 1
        
        # Check if campaign directory exists
        if not campaign_dir.exists():
            print(f"Error: Campaign directory not found: {campaign_dir}")
            print("Hint: Create the campaign first with 'pkdpipe-campaign create'")
            return 1
        
        # Check for campaign state
        state_file = campaign_dir / "campaign_state.json"
        if not state_file.exists():
            print(f"Error: No campaign state found in {campaign_dir}")
            print("Hint: Create the campaign first with 'pkdpipe-campaign create'")
            return 1
        
        # Load the campaign
        campaign = Campaign(str(config_file))
        
        # Check for conflicting flags
        if args.dry_run and args.no_submit:
            print("Error: --dry-run and --no-submit cannot be used together")
            return 1
        
        if args.variant:
            # Submit specific variant
            if args.variant not in [v.name for v in campaign.config.variants]:
                print(f"Error: Unknown variant '{args.variant}'")
                available = [v.name for v in campaign.config.variants]
                print(f"Available variants: {', '.join(available)}")
                return 1
            
            success = campaign.submit_variant(args.variant, dry_run=args.dry_run, no_submit=args.no_submit)
            if success:
                if args.dry_run:
                    print(f"✓ Would submit variant '{args.variant}'")
                elif args.no_submit:
                    print(f"✓ Files created for variant '{args.variant}' (not submitted)")
                else:
                    print(f"✓ Successfully submitted variant '{args.variant}'")
                return 0
            else:
                print(f"✗ Failed to process variant '{args.variant}'")
                return 1
        else:
            # Submit all ready variants
            max_subs = args.max_submissions if hasattr(args, 'max_submissions') else None
            submitted = campaign.submit_ready_variants(
                max_submissions=max_subs, 
                dry_run=args.dry_run,
                no_submit=args.no_submit
            )
            
            if submitted > 0:
                if args.dry_run:
                    print(f"✓ Would submit {submitted} variant(s)")
                elif args.no_submit:
                    print(f"✓ Created files for {submitted} variant(s) (not submitted)")
                else:
                    print(f"✓ Successfully submitted {submitted} variant(s)")
                return 0
            else:
                print("No variants were ready for submission")
                return 0
                
    except Exception as e:
        print(f"Error submitting campaign: {e}")
        return 1


def status_campaign(args) -> int:
    """Show campaign status."""
    try:
        campaign_dir = Path(args.campaign_dir)
        if not campaign_dir.exists():
            print(f"Error: Campaign directory not found: {campaign_dir}")
            return 1
        
        # Find the original config file
        config_files = list(campaign_dir.glob("*.yaml")) + list(campaign_dir.glob("*.yml"))
        if not config_files:
            print(f"Error: No YAML configuration file found in {campaign_dir}")
            return 1
        
        config_file = config_files[0]
        campaign = Campaign(str(config_file))
        
        # Update status from SLURM
        if not args.no_update:
            print("Updating status from SLURM...")
            campaign.update_status()
        
        # Generate and print report
        report = campaign.generate_report()
        print(report)
        
        # Watch mode
        if args.watch:
            import time
            try:
                while True:
                    time.sleep(args.watch)
                    print("\n" + "="*80)
                    print(f"Status update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print("="*80)
                    campaign.update_status()
                    report = campaign.generate_report()
                    print(report)
            except KeyboardInterrupt:
                print("\nWatching stopped.")
        
        return 0
        
    except Exception as e:
        print(f"Error getting campaign status: {e}")
        return 1


def list_campaigns(args) -> int:
    """List available campaigns."""
    try:
        # Look for campaigns in default directory and specified directories
        search_dirs = [Path(".")]
        if hasattr(args, 'search_dir') and args.search_dir:
            search_dirs.append(Path(args.search_dir))
        
        campaigns_found = []
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            # Look for campaign state files
            for state_file in search_dir.rglob("campaign_state.json"):
                campaign_dir = state_file.parent
                campaigns_found.append(campaign_dir)
        
        if not campaigns_found:
            print("No campaigns found.")
            return 0
        
        print(f"Found {len(campaigns_found)} campaign(s):")
        print("-" * 60)
        
        for campaign_dir in campaigns_found:
            try:
                # Try to load campaign info
                config_files = list(campaign_dir.glob("*.yaml")) + list(campaign_dir.glob("*.yml"))
                if config_files:
                    config_file = config_files[0]
                    campaign = Campaign(str(config_file))
                    summary = campaign.get_summary()
                    
                    print(f"Name: {summary['name']}")
                    print(f"Path: {campaign_dir}")
                    print(f"Status: {summary['status']}")
                    print(f"Progress: {summary['completed']}/{summary['total_variants']} "
                          f"({summary['completion_rate']:.1%})")
                    print()
                    
            except Exception as e:
                print(f"Error loading campaign in {campaign_dir}: {e}")
                print()
        
        return 0
        
    except Exception as e:
        print(f"Error listing campaigns: {e}")
        return 1


def main():
    """Main entry point for campaign CLI commands."""
    parser = argparse.ArgumentParser(
        description="pkdpipe campaign management CLI",
        prog="pkdpipe-campaign"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new campaign')
    create_parser.add_argument('config', help='Path to campaign YAML configuration file')
    create_parser.add_argument('--validate-only', action='store_true',
                              help='Only validate configuration without creating campaign')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit campaign simulations')
    submit_parser.add_argument('campaign_dir', help='Path to campaign directory')
    submit_parser.add_argument('--variant', help='Submit specific variant (optional)')
    submit_parser.add_argument('--dry-run', action='store_true',
                              help='Show what would be submitted without doing anything')
    submit_parser.add_argument('--no-submit', action='store_true',
                              help='Create all simulation files and directories but do not submit jobs')
    submit_parser.add_argument('--max-submissions', type=int,
                              help='Maximum number of variants to submit')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show campaign status')
    status_parser.add_argument('campaign_dir', help='Path to campaign directory')
    status_parser.add_argument('--no-update', action='store_true',
                              help='Do not update status from SLURM')
    status_parser.add_argument('--watch', type=int, metavar='SECONDS',
                              help='Watch mode: update status every N seconds')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available campaigns')
    list_parser.add_argument('--search-dir', help='Additional directory to search for campaigns')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate function
    if args.command == 'create':
        return create_campaign(args)
    elif args.command == 'submit':
        return submit_campaign(args)
    elif args.command == 'status':
        return status_campaign(args)
    elif args.command == 'list':
        return list_campaigns(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
