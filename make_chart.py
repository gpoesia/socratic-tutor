# import json
import pickle
import altair

chart = {"data": {"values": [{"env_steps": 1, "success": 2},
                            {"env_steps": 2, "success": 3}]},
         "mark": "line",
         "encoding": {
          "x": {"field": "env_steps", "type": "quantitative"},
          "y": {"field": "success", "type": "quantitative"}
         }
        }

with open("output_ec2/results.pkl", "rb") as f:
    r = pickle.load(f)

values = []
for p in r:
    values.append({"env_steps": p["n_steps"], "success": p["success_rate"]})

chart["data"]["values"] = values

# with open("test_chart.json", "w") as f:
#     json.dump(chart, f)

alt_chart = altair.Chart.from_dict(chart)
alt_chart.show()
