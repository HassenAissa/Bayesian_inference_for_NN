    plotter.plot_decision_boundaries(granularity=0.0001, un_zoom_level=1)
    plotter.plot_uncertainty_area(uncertainty_threshold=0.9, un_zoom_level=1)
    plotter.compare_prediction_to_target()
    plotter.confusion_matrix()