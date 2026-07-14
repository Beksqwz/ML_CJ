# Stage 19G seasonal robustness

{
  "windows": {
    "autumn": {
      "start": "2024-09-30 19:00:00",
      "end": "2024-10-07 18:00:00",
      "hours": 168
    },
    "winter": {
      "start": "2025-01-01 00:00:00",
      "end": "2025-01-07 23:00:00",
      "hours": 168
    },
    "spring": {
      "start": "2025-04-01 00:00:00",
      "end": "2025-04-07 23:00:00",
      "hours": 168
    }
  },
  "metrics": {
    "autumn": {
      "catboost": {
        "natural_prevalence": 0.0018931211597542242,
        "pr_auc": 0.005857766591160944,
        "roc_auc": 0.6645427149267091,
        "precision_at_10": 0.0,
        "precision_at_20": 0.008035714285714287,
        "precision_at_50": 0.009285714285714286,
        "recall_at_1pct": 0.042723149866007004,
        "recall_at_5pct": 0.2911116000401715,
        "recall_at_10pct": 0.3409273596773597,
        "lift_at_1pct": 4.238136466707895,
        "lift_at_5pct": 5.804677532459299,
        "lift_at_10pct": 3.4075560785888244
      },
      "baseline": {
        "natural_prevalence": 0.0018931211597542242,
        "pr_auc": 0.00863666988065382,
        "roc_auc": 0.607293498336625,
        "precision_at_10": 0.02857142857142857,
        "precision_at_20": 0.021428571428571425,
        "precision_at_50": 0.014285714285714287,
        "recall_at_1pct": 0.08983430590573446,
        "recall_at_5pct": 0.24473960232888806,
        "recall_at_10pct": 0.33418132793132793,
        "lift_at_1pct": 8.91156314584886,
        "lift_at_5pct": 4.880033879603155,
        "lift_at_10pct": 3.3401297461750863
      },
      "rows": 666624,
      "timestamps": 168
    },
    "winter": {
      "catboost": {
        "natural_prevalence": 0.001068068356374808,
        "pr_auc": 0.0017093093299893516,
        "roc_auc": 0.5735249978537502,
        "precision_at_10": 0.0,
        "precision_at_20": 0.0,
        "precision_at_50": 0.0,
        "recall_at_1pct": 0.0,
        "recall_at_5pct": 0.12854030501089322,
        "recall_at_10pct": 0.2660675381263617,
        "lift_at_1pct": 0.0,
        "lift_at_5pct": 2.563054926046354,
        "lift_at_10pct": 2.659334990643333
      },
      "baseline": {
        "natural_prevalence": 0.001068068356374808,
        "pr_auc": 0.0015936623570309088,
        "roc_auc": 0.5148332793340039,
        "precision_at_10": 0.0,
        "precision_at_20": 0.0,
        "precision_at_50": 0.0,
        "recall_at_1pct": 0.0,
        "recall_at_5pct": 0.041607013175640625,
        "recall_at_10pct": 0.2666199813258636,
        "lift_at_1pct": 0.0,
        "lift_at_5pct": 0.829631297894181,
        "lift_at_10pct": 2.664856639549187
      },
      "rows": 666624,
      "timestamps": 168
    },
    "spring": {
      "catboost": {
        "natural_prevalence": 0.001327584965437788,
        "pr_auc": 0.003138081232047388,
        "roc_auc": 0.6594203622476268,
        "precision_at_10": 0.0017857142857142859,
        "precision_at_20": 0.0011904761904761906,
        "precision_at_50": 0.006190476190476191,
        "recall_at_1pct": 0.059226190476190474,
        "recall_at_5pct": 0.15707199546485257,
        "recall_at_10pct": 0.26530612244897955,
        "lift_at_1pct": 5.875238095238096,
        "lift_at_5pct": 3.131968231178569,
        "lift_at_10pct": 2.651724669716753
      },
      "baseline": {
        "natural_prevalence": 0.001327584965437788,
        "pr_auc": 0.007091728992996992,
        "roc_auc": 0.6065305932527145,
        "precision_at_10": 0.014285714285714287,
        "precision_at_20": 0.0071428571428571435,
        "precision_at_50": 0.005714285714285715,
        "recall_at_1pct": 0.047704081632653064,
        "recall_at_5pct": 0.23410572562358276,
        "recall_at_10pct": 0.3149730725623583,
        "lift_at_1pct": 4.732244897959184,
        "lift_at_5pct": 4.667997584293349,
        "lift_at_10pct": 3.148143959514956
      },
      "rows": 666624,
      "timestamps": 168
    }
  },
  "comparison": {
    "autumn": {
      "natural_prevalence": {
        "catboost": 0.0018931211597542242,
        "baseline": 0.0018931211597542242,
        "absolute_difference": 0.0,
        "relative_improvement_pct": 0.0,
        "winner": "tie"
      },
      "pr_auc": {
        "catboost": 0.005857766591160944,
        "baseline": 0.00863666988065382,
        "absolute_difference": -0.002778903289492876,
        "relative_improvement_pct": -32.1756339873269,
        "winner": "baseline"
      },
      "roc_auc": {
        "catboost": 0.6645427149267091,
        "baseline": 0.607293498336625,
        "absolute_difference": 0.057249216590084084,
        "relative_improvement_pct": 9.426943767204737,
        "winner": "catboost"
      },
      "precision_at_10": {
        "catboost": 0.0,
        "baseline": 0.02857142857142857,
        "absolute_difference": -0.02857142857142857,
        "relative_improvement_pct": -100.0,
        "winner": "baseline"
      },
      "precision_at_20": {
        "catboost": 0.008035714285714287,
        "baseline": 0.021428571428571425,
        "absolute_difference": -0.013392857142857139,
        "relative_improvement_pct": -62.499999999999986,
        "winner": "baseline"
      },
      "precision_at_50": {
        "catboost": 0.009285714285714286,
        "baseline": 0.014285714285714287,
        "absolute_difference": -0.005000000000000001,
        "relative_improvement_pct": -35.0,
        "winner": "baseline"
      },
      "recall_at_1pct": {
        "catboost": 0.042723149866007004,
        "baseline": 0.08983430590573446,
        "absolute_difference": -0.04711115603972746,
        "relative_improvement_pct": -52.442277551698844,
        "winner": "baseline"
      },
      "recall_at_5pct": {
        "catboost": 0.2911116000401715,
        "baseline": 0.24473960232888806,
        "absolute_difference": 0.04637199771128345,
        "relative_improvement_pct": 18.947484293517576,
        "winner": "catboost"
      },
      "recall_at_10pct": {
        "catboost": 0.3409273596773597,
        "baseline": 0.33418132793132793,
        "absolute_difference": 0.006746031746031778,
        "relative_improvement_pct": 2.0186740497416547,
        "winner": "catboost"
      },
      "lift_at_1pct": {
        "catboost": 4.238136466707895,
        "baseline": 8.91156314584886,
        "absolute_difference": -4.673426679140964,
        "relative_improvement_pct": -52.442277551698844,
        "winner": "baseline"
      },
      "lift_at_5pct": {
        "catboost": 5.804677532459299,
        "baseline": 4.880033879603155,
        "absolute_difference": 0.9246436528561439,
        "relative_improvement_pct": 18.947484293517572,
        "winner": "catboost"
      },
      "lift_at_10pct": {
        "catboost": 3.4075560785888244,
        "baseline": 3.3401297461750863,
        "absolute_difference": 0.06742633241373808,
        "relative_improvement_pct": 2.0186740497416493,
        "winner": "catboost"
      }
    },
    "winter": {
      "natural_prevalence": {
        "catboost": 0.001068068356374808,
        "baseline": 0.001068068356374808,
        "absolute_difference": 0.0,
        "relative_improvement_pct": 0.0,
        "winner": "tie"
      },
      "pr_auc": {
        "catboost": 0.0017093093299893516,
        "baseline": 0.0015936623570309088,
        "absolute_difference": 0.00011564697295844283,
        "relative_improvement_pct": 7.256679713129466,
        "winner": "catboost"
      },
      "roc_auc": {
        "catboost": 0.5735249978537502,
        "baseline": 0.5148332793340039,
        "absolute_difference": 0.058691718519746305,
        "relative_improvement_pct": 11.40014075929023,
        "winner": "catboost"
      },
      "precision_at_10": {
        "catboost": 0.0,
        "baseline": 0.0,
        "absolute_difference": 0.0,
        "relative_improvement_pct": null,
        "winner": "tie"
      },
      "precision_at_20": {
        "catboost": 0.0,
        "baseline": 0.0,
        "absolute_difference": 0.0,
        "relative_improvement_pct": null,
        "winner": "tie"
      },
      "precision_at_50": {
        "catboost": 0.0,
        "baseline": 0.0,
        "absolute_difference": 0.0,
        "relative_improvement_pct": null,
        "winner": "tie"
      },
      "recall_at_1pct": {
        "catboost": 0.0,
        "baseline": 0.0,
        "absolute_difference": 0.0,
        "relative_improvement_pct": null,
        "winner": "tie"
      },
      "recall_at_5pct": {
        "catboost": 0.12854030501089322,
        "baseline": 0.041607013175640625,
        "absolute_difference": 0.0869332918352526,
        "relative_improvement_pct": 208.93903503303824,
        "winner": "catboost"
      },
      "recall_at_10pct": {
        "catboost": 0.2660675381263617,
        "baseline": 0.2666199813258636,
        "absolute_difference": -0.0005524431995019286,
        "relative_improvement_pct": -0.2072024747562828,
        "winner": "baseline"
      },
      "lift_at_1pct": {
        "catboost": 0.0,
        "baseline": 0.0,
        "absolute_difference": 0.0,
        "relative_improvement_pct": null,
        "winner": "tie"
      },
      "lift_at_5pct": {
        "catboost": 2.563054926046354,
        "baseline": 0.829631297894181,
        "absolute_difference": 1.733423628152173,
        "relative_improvement_pct": 208.93903503303827,
        "winner": "catboost"
      },
      "lift_at_10pct": {
        "catboost": 2.659334990643333,
        "baseline": 2.664856639549187,
        "absolute_difference": -0.0055216489058538265,
        "relative_improvement_pct": -0.20720247475631268,
        "winner": "baseline"
      }
    },
    "spring": {
      "natural_prevalence": {
        "catboost": 0.001327584965437788,
        "baseline": 0.001327584965437788,
        "absolute_difference": 0.0,
        "relative_improvement_pct": 0.0,
        "winner": "tie"
      },
      "pr_auc": {
        "catboost": 0.003138081232047388,
        "baseline": 0.007091728992996992,
        "absolute_difference": -0.003953647760949605,
        "relative_improvement_pct": -55.75012475594866,
        "winner": "baseline"
      },
      "roc_auc": {
        "catboost": 0.6594203622476268,
        "baseline": 0.6065305932527145,
        "absolute_difference": 0.05288976899491227,
        "relative_improvement_pct": 8.720049669922494,
        "winner": "catboost"
      },
      "precision_at_10": {
        "catboost": 0.0017857142857142859,
        "baseline": 0.014285714285714287,
        "absolute_difference": -0.0125,
        "relative_improvement_pct": -87.5,
        "winner": "baseline"
      },
      "precision_at_20": {
        "catboost": 0.0011904761904761906,
        "baseline": 0.0071428571428571435,
        "absolute_difference": -0.005952380952380953,
        "relative_improvement_pct": -83.33333333333334,
        "winner": "baseline"
      },
      "precision_at_50": {
        "catboost": 0.006190476190476191,
        "baseline": 0.005714285714285715,
        "absolute_difference": 0.00047619047619047554,
        "relative_improvement_pct": 8.33333333333332,
        "winner": "catboost"
      },
      "recall_at_1pct": {
        "catboost": 0.059226190476190474,
        "baseline": 0.047704081632653064,
        "absolute_difference": 0.01152210884353741,
        "relative_improvement_pct": 24.153297682709436,
        "winner": "catboost"
      },
      "recall_at_5pct": {
        "catboost": 0.15707199546485257,
        "baseline": 0.23410572562358276,
        "absolute_difference": -0.07703373015873019,
        "relative_improvement_pct": -32.905530163150424,
        "winner": "baseline"
      },
      "recall_at_10pct": {
        "catboost": 0.26530612244897955,
        "baseline": 0.3149730725623583,
        "absolute_difference": -0.04966695011337874,
        "relative_improvement_pct": -15.768633715044226,
        "winner": "baseline"
      },
      "lift_at_1pct": {
        "catboost": 5.875238095238096,
        "baseline": 4.732244897959184,
        "absolute_difference": 1.142993197278912,
        "relative_improvement_pct": 24.153297682709454,
        "winner": "catboost"
      },
      "lift_at_5pct": {
        "catboost": 3.131968231178569,
        "baseline": 4.667997584293349,
        "absolute_difference": -1.5360293531147806,
        "relative_improvement_pct": -32.90553016315041,
        "winner": "baseline"
      },
      "lift_at_10pct": {
        "catboost": 2.651724669716753,
        "baseline": 3.148143959514956,
        "absolute_difference": -0.49641928979820316,
        "relative_improvement_pct": -15.768633715044212,
        "winner": "baseline"
      }
    }
  },
  "top10_stability": {
    "autumn": {
      "unique_top10_segments": 11,
      "overlap_with_previous": null,
      "jaccard_with_previous": null,
      "permanently_dominant_segments": 9,
      "new_top10_segments": 11
    },
    "winter": {
      "unique_top10_segments": 11,
      "overlap_with_previous": 7,
      "jaccard_with_previous": 0.4666666666666667,
      "permanently_dominant_segments": 9,
      "new_top10_segments": 4
    },
    "spring": {
      "unique_top10_segments": 12,
      "overlap_with_previous": 8,
      "jaccard_with_previous": 0.5333333333333333,
      "permanently_dominant_segments": 8,
      "new_top10_segments": 4
    }
  },
  "seasonal_analysis": {
    "best_roc_auc": "autumn",
    "worst_roc_auc": "winter",
    "catboost_consistent_over_baseline": true
  }
}