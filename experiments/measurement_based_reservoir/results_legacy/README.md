# Legacy pre-V3 artifacts

Everything in this directory is preserved historical evidence generated before
the `cv_mb_qrc` namespace and V3 feature/statistics corrections. It must not be
mixed with `results_ci`, `results_development` or `results_publication`, and it
does not validate current source code.

The archived build products are retained for recoverability rather than copied
into ordinary V3 build output:

```text
30794ac8cbbd999432ff4cd8ca5dcc6d8b40f17a153dd85b0345f42006c4650b  build/cv_mb_qrc-0.1.0-py3-none-any.whl
878af02f52978288b7ab589619a78670a37a8e5989d336b2ee434b14acf45ba6  build/cv_mb_qrc-0.1.0.tar.gz
```

Create current distributions with `python -m build`, which writes to the ignored
top-level `dist/` directory. Release binaries should be attached to a release,
not committed beside source.
