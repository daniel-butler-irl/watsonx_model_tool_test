#!/usr/bin/env python3
"""
Tests for HTML report generation functionality.

This module contains tests for the HTML report generator.
"""

import os
import tempfile

from watsonx_tool_tester.utils.html_generator import HTMLReportGenerator


class TestHTMLReportGenerator:
    """Test class for HTML report generator."""

    def test_generate_html_report_basic(self):
        """Test basic HTML report generation."""
        generator = HTMLReportGenerator()

        # Sample test results
        results = [
            {
                "model": "mock-test-model-1",
                "tool_call_support": True,
                "handles_response": True,
                "details": "Success",
                "response_times": {
                    "tool_call_time": 1.0,
                    "response_processing_time": 0.5,
                    "total_time": 1.5,
                },
            },
            {
                "model": "mock-test-model-2",
                "tool_call_support": True,
                "handles_response": False,
                "details": "Called tool but ignored result",
                "response_times": {
                    "tool_call_time": 2.0,
                    "response_processing_time": 1.0,
                    "total_time": 3.0,
                },
            },
            {
                "model": "mock-test-model-3",
                "tool_call_support": False,
                "handles_response": False,
                "details": "No tool calling support",
                "response_times": {
                    "tool_call_time": None,
                    "response_processing_time": None,
                    "total_time": None,
                },
            },
        ]

        # Sample summary
        summary = {
            "total_count": 3,
            "supported_count": 2,
            "handles_response_count": 1,
            "avg_tool_time": 1.5,
            "avg_response_time": 0.75,
            "avg_total_time": 2.25,
            "fastest_model": {
                "model": "mock-test-model-1",
                "time": 1.5,
            },
        }

        # Generate HTML report
        html_content = generator.generate_html_report(results, summary)

        # Basic structure checks
        assert "<!DOCTYPE html>" in html_content
        assert '<html lang="en">' in html_content
        assert "WatsonX Tool Test Report" in html_content
        assert "mock-test-model-1" in html_content
        assert "test-model-2" in html_content
        assert "test-model-3" in html_content

        # Check for interactive elements
        assert 'id="model-filter"' in html_content
        assert 'id="support-filter"' in html_content
        assert 'id="handling-filter"' in html_content

        # Check for styling
        assert "<style>" in html_content
        assert "var(--primary-color)" in html_content
        assert "var(--success-color)" in html_content

        # Check for JavaScript
        assert "<script>" in html_content
        assert "initializeFilters" in html_content

    def test_generate_html_report_with_reliability(self):
        """Test HTML report generation with reliability data."""
        generator = HTMLReportGenerator()

        # Sample test results with reliability data
        results = [
            {
                "model": "reliable-model",
                "tool_call_support": True,
                "handles_response": True,
                "details": "Consistently successful",
                "response_times": {
                    "tool_call_time": 1.0,
                    "response_processing_time": 0.5,
                    "total_time": 1.5,
                },
                "reliability": {
                    "iterations": 5,
                    "is_reliable": True,
                    "tool_call_success_rate": 1.0,
                    "response_handling_success_rate": 1.0,
                },
            },
            {
                "model": "unreliable-model",
                "tool_call_support": True,
                "handles_response": False,
                "details": "Inconsistent behavior",
                "response_times": {
                    "tool_call_time": 2.0,
                    "response_processing_time": 1.0,
                    "total_time": 3.0,
                },
                "reliability": {
                    "iterations": 5,
                    "is_reliable": False,
                    "tool_call_success_rate": 0.8,
                    "response_handling_success_rate": 0.4,
                },
            },
        ]

        # Sample summary with reliability
        summary = {
            "total_count": 2,
            "supported_count": 2,
            "handles_response_count": 1,
            "avg_tool_time": 1.5,
            "avg_response_time": 0.75,
            "avg_total_time": 2.25,
            "reliability": {
                "iterations": 5,
                "reliable_count": 1,
                "unreliable_count": 1,
                "avg_tool_success_rate": 0.9,
                "avg_response_success_rate": 0.7,
            },
        }

        # Generate HTML report
        html_content = generator.generate_html_report(results, summary)

        # Check for reliability-specific content
        assert "Reliability (5x)" in html_content
        assert "✅ 100%/100%" in html_content
        assert "⚠️ 80%/40%" in html_content
        assert "Reliability Assessment" in html_content
        assert "Reliable Models" in html_content
        assert "Unreliable Models" in html_content

    def test_generate_html_report_with_config(self):
        """Test HTML report generation with configuration info."""
        generator = HTMLReportGenerator()

        # Sample results and summary
        results = [
            {
                "model": "test-model",
                "tool_call_support": True,
                "handles_response": True,
                "details": "Success",
                "response_times": {
                    "tool_call_time": 1.0,
                    "response_processing_time": 0.5,
                    "total_time": 1.5,
                },
            }
        ]

        summary = {
            "total_count": 1,
            "supported_count": 1,
            "handles_response_count": 1,
            "avg_tool_time": 1.0,
            "avg_response_time": 0.5,
            "avg_total_time": 1.5,
        }

        # Configuration info
        config = {
            "iterations": 10,
            "client_type": "watsonx",
            "total_models": 1,
        }

        # Generate HTML report
        html_content = generator.generate_html_report(results, summary, config)

        # Check for configuration content
        assert "Iterations: 10" in html_content
        assert "Client: watsonx" in html_content
        assert "Models Tested: 1" in html_content

    def test_save_html_report(self):
        """Test saving HTML report to file."""
        generator = HTMLReportGenerator()

        # Sample HTML content
        html_content = "<!DOCTYPE html><html><head><title>Test</title></head><body><h1>Test Report</h1></body></html>"

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False
        ) as f:
            temp_path = f.name

        try:
            # Save HTML report
            generator.save_html_report(html_content, temp_path)

            # Verify file was created and contains expected content
            assert os.path.exists(temp_path)

            with open(temp_path, "r", encoding="utf-8") as f:
                saved_content = f.read()

            assert saved_content == html_content

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_html_report_creates_directory(self):
        """Test that save_html_report creates directory if needed."""
        generator = HTMLReportGenerator()

        # Sample HTML content
        html_content = "<!DOCTYPE html><html><head><title>Test</title></head><body><h1>Test Report</h1></body></html>"

        # Create temporary directory path
        with tempfile.TemporaryDirectory() as temp_dir:
            subdir_path = os.path.join(temp_dir, "subdir", "report.html")

            # Save HTML report (should create subdir)
            generator.save_html_report(html_content, subdir_path)

            # Verify directory was created and file exists
            assert os.path.exists(subdir_path)

            with open(subdir_path, "r", encoding="utf-8") as f:
                saved_content = f.read()

            assert saved_content == html_content

    def test_css_styles_included(self):
        """Test that CSS styles are properly included."""
        generator = HTMLReportGenerator()

        # Get CSS styles
        css_content = generator._get_css_styles()

        # Check for key CSS variables and classes
        assert "--primary-color" in css_content
        assert "--success-color" in css_content
        assert "--warning-color" in css_content
        assert "--error-color" in css_content
        assert ".results-table" in css_content
        assert ".summary-card" in css_content
        assert ".support-indicator" in css_content
        assert ".reliability-badge" in css_content

        # Check for responsive design
        assert "@media (max-width: 768px)" in css_content

    def test_javascript_functionality(self):
        """Test that JavaScript functions are included."""
        generator = HTMLReportGenerator()

        # Get JavaScript content
        js_content = generator._get_javascript()

        # Check for key functions
        assert "initializeFilters" in js_content
        assert "initializeCollapsibles" in js_content
        assert "initializeTableSorting" in js_content
        assert "filterTable" in js_content
        assert "sortTable" in js_content

        # Check for event listeners
        assert "addEventListener" in js_content
        assert "DOMContentLoaded" in js_content

    def test_empty_results_handling(self):
        """Test handling of empty results."""
        generator = HTMLReportGenerator()

        # Empty results
        results = []
        summary = {
            "total_count": 0,
            "supported_count": 0,
            "handles_response_count": 0,
            "avg_tool_time": 0,
            "avg_response_time": 0,
            "avg_total_time": 0,
        }

        # Generate HTML report
        html_content = generator.generate_html_report(results, summary)

        # Should still generate valid HTML
        assert "<!DOCTYPE html>" in html_content
        assert "WatsonX Tool Test Report" in html_content
        assert "Total Models" in html_content
        assert "0" in html_content  # Should show 0 models tested
