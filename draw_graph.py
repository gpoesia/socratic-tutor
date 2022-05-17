import sys
import collections
import pickle
import matplotlib.pyplot as plt

def load_data(path):
    data_points = {}

    with open(path, 'rb') as f:
        r = pickle.load(f)

    for p in r:
        algorithm, domain = p['name'], p['domain']
        if (algorithm, domain) not in data_points:
            data_points[algorithm, domain] = ([], [])
        data_points[algorithm, domain][0].append(p['n_steps'])
        data_points[algorithm, domain][1].append(p['success_rate'])
    
    return data_points

"""
def make_plot(data: list[dict], plot_id: str):
    with open(os.path.join('vega-lite', plot_id + '.json')) as f:
        plot_spec = json.load(f)
    plot_spec['data'] = {'values': data}
    return altair.Chart.from_dict(plot_spec)
"""
def make_plot(data_points, save):
    for key, val in data_points.items():
        plt.plot(*val)
        if save:
            plt.savefig('-'.join(key)+'.svg')
            plt.savefig('-'.join(key)+'.png')
        else:
            plt.show()
        plt.clf()

if __name__ == "__main__":
    path = sys.argv[1]
    save = sys.argv[2]
    data_points = load_data(path)
    make_plot(data_points, save)
