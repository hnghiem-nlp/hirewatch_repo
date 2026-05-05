This directory is the landing zone for the v4 trial-results bundle that
accompanies the paper.

After unzipping the bundle, the layout under this directory should be:

  results/interview_v4/{condition}/{model_slug}/temp_{T}/persona{P}/all_results.json
  results/thinking_ablation/{condition}/{model_slug}/temp_0.0/personaB/*.json

The notebooks under ../notebooks/ read directly from these paths.
