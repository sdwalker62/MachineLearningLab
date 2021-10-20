# install jupyterlab extensions
jupyter labextension install --no-build @krassowski/jupyterlab-lsp
jupyter labextension install --no-build @jupyterlab/debugger
jupyter labextension install --no-build @wallneradam/output_auto_scroll
jupyter lab build --dev-build=False --minimize=False