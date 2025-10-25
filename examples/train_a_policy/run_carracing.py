"""
Wrapper to run CarRacing with software rendering
"""
import os
import sys

# Set environment variables before any imports
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['GALLIUM_DRIVER'] = 'llvmpipe'

# Disable pyglet shadow window that causes issues
import pyglet
pyglet.options['shadow_window'] = False

# Now run the training
from pls.workflows.execute_workflow import train

if __name__ == "__main__":
    cwd = os.path.dirname(__file__)
    config_file = os.path.join(cwd, "carracing/no_shield/seed1/config.json")
    train(config_file)
