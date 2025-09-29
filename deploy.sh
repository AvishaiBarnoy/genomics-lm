#!/bin/sh

# Deploying for my DNA LLM
# 
# Usage: add custom message after ./deploy 

# If a command fails then the deploy stops
set -e

printf "\033[0;32mDeploying updates to GitHub...\033[0m\n"

# Go To Public folder
pwd

# Add changes to git.
git add .

# Commit changes.
msg="uploading updates $(date)"	# default deployment message
if [ -n "$*" ]; then
		msg="$*"
	fi
	echo $msg
	git commit -m "$msg"

	# Push source and build repos.
	current_branch=$(git rev-parse --abbrev-ref HEAD)
	git push origin "$current_branch"
