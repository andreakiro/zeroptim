.PHONY: setup
setup:
	pip install -r requirements.txt
	make package

.PHONY: package
package:
	pip uninstall -y zeroptim
	rm -rf zeroptim/zeroptim.egg-info
	pip install -e zeroptim

.PHONY: freeze
freeze:
	pip freeze > requirements.txt

.PHONY: clean
clean:
	rm -rf wandb
