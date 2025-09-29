"""
Logging utilities for the ControlMAD project.

This module contains custom logging handlers and utilities used across the project.
"""

import logging
import re


class HTMLFileHandler(logging.FileHandler):
    """Custom logging handler that converts ANSI color codes to HTML."""
    
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        super().__init__(filename, mode, encoding, delay)
        self.html_started = False
    
    def emit(self, record):
        if not self.html_started:
            # Write HTML header
            html_header = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Debate Log</title>
    <style>
        body { font-family: 'Courier New', monospace; background-color: #ffffff; color: #000000; margin: 20px; }
        .log-entry { margin: 5px 0; white-space: pre-wrap; }
        .red { color: #d32f2f; }
        .orange { color: #f57c00; }
        .blue { color: #1976d2; }
        .green { color: #388e3c; }
        .yellow { color: #f9a825; }
        .purple { color: #7b1fa2; }
        .cyan { color: #0097a7; }
        .white { color: #000000; }
        .bold { font-weight: bold; }
        .timestamp { color: #666; }
        .level { font-weight: bold; }
        .info { color: #1976d2; }
        .warning { color: #f57c00; }
        .error { color: #d32f2f; }
        .debug { color: #666; }
    </style>
</head>
<body>
"""
            self.stream.write(html_header)
            self.html_started = True
        
        try:
            msg = self.format(record)
            
            # Convert ANSI color codes to HTML
            html_msg = self._ansi_to_html(msg)
            
            # Add HTML wrapper
            html_entry = f'<div class="log-entry">{html_msg}</div>\n'
            self.stream.write(html_entry)
            self.stream.flush()
        except Exception:
            self.handleError(record)
    
    def _ansi_to_html(self, text):
        """Convert ANSI color codes to HTML spans."""
        # ANSI color mappings
        color_map = {
            '\033[31m': '<span class="red">',  # Red
            '\033[33m': '<span class="orange">',  # Orange/Yellow
            '\033[34m': '<span class="blue">',  # Blue
            '\033[32m': '<span class="green">',  # Green
            '\033[35m': '<span class="purple">',  # Magenta
            '\033[36m': '<span class="cyan">',  # Cyan
            '\033[37m': '<span class="white">',  # White
            '\033[1m': '<span class="bold">',  # Bold
            '\033[0m': '</span>',  # Reset
        }
        
        # Replace ANSI codes with HTML spans
        for ansi_code, html_span in color_map.items():
            text = text.replace(ansi_code, html_span)
        
        # Handle multiple resets (close all spans)
        text = re.sub(r'</span>(?=</span>)', '', text)
        
        # Escape HTML special characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Restore our HTML spans
        text = text.replace('&lt;span class="red"&gt;', '<span class="red">')
        text = text.replace('&lt;span class="orange"&gt;', '<span class="orange">')
        text = text.replace('&lt;span class="blue"&gt;', '<span class="blue">')
        text = text.replace('&lt;span class="green"&gt;', '<span class="green">')
        text = text.replace('&lt;span class="purple"&gt;', '<span class="purple">')
        text = text.replace('&lt;span class="cyan"&gt;', '<span class="cyan">')
        text = text.replace('&lt;span class="white"&gt;', '<span class="white">')
        text = text.replace('&lt;span class="bold"&gt;', '<span class="bold">')
        text = text.replace('&lt;/span&gt;', '</span>')
        
        return text
    
    def close(self):
        if self.html_started:
            self.stream.write('</body>\n</html>\n')
        super().close()
