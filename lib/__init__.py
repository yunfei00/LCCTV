"""Keep the top-level lib package lightweight.

This package is imported as part of `import lib.inference`, so avoid importing
heavy submodules here. Compatibility shims live in dedicated modules such as
`lib.lcctv`.
"""
