dependencies: ## create cache
	@mkdir -p .cache;
	date +"inits_+%FT%T%z"
	@for t in "episcape" "evaluate" "misc" "models"; do \
		mv $$t/__init__.py .cache ; \
		echo import * ; \
	done

