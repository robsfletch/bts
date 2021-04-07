.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = bts
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

raw_data = data/raw
interim_data = data/interim
processed_data = data/processed

raw_teams = $(raw_data)/Teams.csv
raw_people = $(raw_data)/People.csv
raw_ratings538 = $(raw_data)/mlb_elo.csv

game_logs = $(interim_data)/game_logs.pkl
events = $(interim_data)/events.pkl
adj_events = $(interim_data)/adj_events.pkl
rosters = $(interim_data)/rosters.pkl
ratings538 = $(interim_data)/ratings538.pkl
people = $(interim_data)/people.pkl
teams = $(interim_data)/teams.pkl

batting_games = $(interim_data)/batting_games.pkl
pitching_games = $(interim_data)/pitching_games.pkl
pitching_team_games = $(interim_data)/pitching_team_games.pkl
directory = $(interim_data)/directory.pkl
batting_records = $(interim_data)/batting_records.pkl
batting_records_predict = $(interim_data)/batting_records_predict.pkl
batting_team_records = $(interim_data)/batting_team_records.pkl
batting_team_records_predict = $(interim_data)/batting_team_records_predict.pkl
pitching_records_predict = $(interim_data)/pitching_records_predict.pkl
pitching_records = $(interim_data)/pitching_records.pkl
pitching_team_records = $(interim_data)/pitching_team_records.pkl
pitching_team_records_predict = $(interim_data)/pitching_team_records_predict.pkl
park_records = $(interim_data)/park_records.pkl

panel = $(interim_data)/panel.pkl
merged_data = $(interim_data)/merged_data.pkl
main_data = $(processed_data)/main_data.pkl

main_data_X = $(processed_data)/main_data_X.npy
main_data_Y = $(processed_data)/main_data_Y.npy

predictions = $(processed_data)/main_predictions.pkl
selection = $(processed_data)/main_selection.pkl
selection_data = $(processed_data)/selection_data.pkl

logistic = models/logistic_model.pkl

lineup = $(interim_data)/lineup.pkl
merged_lineup = $(interim_data)/merged_lineup.pkl
lineup_selections = $(interim_data)/lineup_selections.pkl

VPATH = src/data:$\
				src/features:$\
				src/models
#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: $(selection_data)

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py


prediction: $(lineup_selections)

$(lineup_selections): select_from_lineup.py $(merged_lineup) $(logistic)
	$(PYTHON_INTERPRETER) $< $(merged_lineup) $(logistic) $(lineup_selections)

$(merged_lineup): merged_data.py $(lineup) $(batting_games) $(pitching_games) $(pitching_team_games) $(batting_records_predict) $(pitching_records_predict) $(park_records) $(pitching_team_records) $(ratings538)
	$(PYTHON_INTERPRETER) $< $(interim_data) $(lineup) $(merged_lineup)


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
$(selection_data): selection_data.py $(predictions) $(main_data)
	$(PYTHON_INTERPRETER) $< $(interim_data) $(processed_data)

$(predictions): predict_model.py $(main_data) $(logistic)
	$(PYTHON_INTERPRETER) $< $(main_data) $(logistic) $(predictions)

$(logistic): train_model.py $(main_data) modelsetup.py
	$(PYTHON_INTERPRETER) $< $(processed_data) models

$(main_data): main_data.py $(merged_data)
	$(PYTHON_INTERPRETER) $< $(interim_data) $(processed_data)

$(merged_data): merged_data.py $(panel) $(batting_games) $(pitching_games) $(pitching_team_games) $(batting_records_predict) $(pitching_records_predict) $(park_records) $(pitching_team_records_predict) $(batting_team_records_predict) $(ratings538)
	$(PYTHON_INTERPRETER) $< $(interim_data) $(panel) $(merged_data)

$(panel): panel.py $(game_logs)
	$(PYTHON_INTERPRETER) $< $(interim_data)

$(pitching_team_games): pitching_team_games.py $(events) $(directory)
	$(PYTHON_INTERPRETER) $< $(interim_data)

$(pitching_games): pitching_games.py $(events)
	$(PYTHON_INTERPRETER) $< $(interim_data)

$(batting_games): batting_games.py $(events)
	$(PYTHON_INTERPRETER) $< $(interim_data)

$(batting_team_records_predict): batting_team_records_predict.py $(batting_team_records)
	$(PYTHON_INTERPRETER) $< $(interim_data)

$(batting_team_records): batting_team_records.py $(events) $(directory)
	$(PYTHON_INTERPRETER) $< $(interim_data)

$(pitching_team_records_predict): pitching_team_records_predict.py $(pitching_team_records)
	$(PYTHON_INTERPRETER) $< $(interim_data)

$(pitching_team_records): pitching_team_records.py $(events) $(directory)
	$(PYTHON_INTERPRETER) $< $(interim_data)

$(pitching_records_predict): pitching_records_predict.py marcel.py $(pitching_records) $(adj_events) $(game_logs) $(people)
	$(PYTHON_INTERPRETER) $< $(interim_data)

$(pitching_records): pitching_records.py $(adj_events) $(directory)
	$(PYTHON_INTERPRETER) $< $(interim_data)

$(batting_records_predict): batting_records_predict.py marcel.py $(batting_records) $(adj_events) $(game_logs) $(people)
	$(PYTHON_INTERPRETER) $< $(interim_data)

$(batting_records): batting_records.py $(adj_events) $(directory)
	$(PYTHON_INTERPRETER) $< $(interim_data)

$(directory): directory.py $(rosters)
	$(PYTHON_INTERPRETER) $< $(interim_data)

$(ratings538): ratings538.py $(raw_ratings538) $(teams)
	$(PYTHON_INTERPRETER) $< $(raw_data) $(interim_data)

$(adj_events): adj_events.py $(events) $(game_logs) $(park_records)
	$(PYTHON_INTERPRETER) $< $(interim_data)

$(events): events.py
	$(PYTHON_INTERPRETER) $< $(raw_data) $(interim_data)

$(park_records): park_records.py $(game_logs)
	$(PYTHON_INTERPRETER) $< $(interim_data)

$(game_logs): game_logs.py
	$(PYTHON_INTERPRETER) $< $(raw_data) $(interim_data)

$(rosters): rosters.py
	$(PYTHON_INTERPRETER) $< $(raw_data) $(interim_data)

$(people): lahman_people.py $(raw_people)
	$(PYTHON_INTERPRETER) $< $(raw_data) $(interim_data)

$(teams): lahman_teams.py $(raw_teams)
	$(PYTHON_INTERPRETER) $< $(raw_data) $(interim_data)


raw_events:
	src/data/import_events.sh

# .FORCE
$(raw_ratings538):
	src/data/import538.sh

# switch to downloading wholde directory
$(raw_people):
	src/data/import_people.sh

$(raw_teams):
	src/data/import_people.sh

.FORCE:


#################################################################################
# Clearing
#################################################################################
clear:
	rm -f $(rosters)
	rm -f $(game_logs)
	rm -f $(events)



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
