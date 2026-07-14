# Stage 19I weight search

{
  "best_two": {
    "weights": {
      "score_catboost_stage19h": 0.8,
      "score_hist_gradient_boosting": 0.19999999999999996,
      "score_logistic_regression": 0.0
    },
    "validation": {
      "pr_auc": 0.28420836316515546,
      "roc_auc": 0.6463727582214999
    },
    "subperiods": [
      {
        "pr_auc": 0.2845265405738443,
        "roc_auc": 0.6519125719932439
      },
      {
        "pr_auc": 0.28951740332669834,
        "roc_auc": 0.6480413192239357
      },
      {
        "pr_auc": 0.28111708898736876,
        "roc_auc": 0.6395699521310609
      }
    ],
    "minimum_subperiod_pr_auc": 0.28111708898736876,
    "std_subperiod_pr_auc": 0.003449611123621081
  },
  "best_three": {
    "weights": {
      "score_catboost_stage19h": 0.7,
      "score_hist_gradient_boosting": 0.2,
      "score_logistic_regression": 0.09999999999999998
    },
    "validation": {
      "pr_auc": 0.28548770964734904,
      "roc_auc": 0.6462083699324375
    },
    "subperiods": [
      {
        "pr_auc": 0.28434654428603806,
        "roc_auc": 0.6519643581893733
      },
      {
        "pr_auc": 0.29025486849300086,
        "roc_auc": 0.6482040792104864
      },
      {
        "pr_auc": 0.2832331163238594,
        "roc_auc": 0.6388044315543272
      }
    ],
    "minimum_subperiod_pr_auc": 0.2832331163238594,
    "std_subperiod_pr_auc": 0.003081360094474641
  },
  "three_accepted": false
}