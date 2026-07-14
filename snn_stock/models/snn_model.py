# snn_stock/models/snn_model.py

import logging
import numpy as np
from ml_genn import Network, Population, Connection
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrateFire, SpikeInput, LeakyIntegrate
from ml_genn.synapses import Delta, Exponential
from ml_genn.connectivity import Dense


def build_snn(input_size, hidden_layers, output_size, neuron_params,
              output_readout="var", algorithm="eprop", max_input_spikes=None,
              recurrent=False):
    """
    Builds a spiking neural network (SNN) using the mlGeNN framework.

    Args:
        input_size: Number of input neurons
        hidden_layers: List of dictionaries defining hidden layer configurations
        output_size: Number of output neurons
        neuron_params: Dictionary of neuron parameters
        output_readout: Readout mechanism for the output layer (e.g. "var",
            "avg_var") - the output layer is always a non-spiking
            leaky integrator whose voltage encodes the prediction
        algorithm: Learning algorithm ('eprop' or 'eventprop')
        max_input_spikes: Size of the input spike buffer (across the batch)
    """
    algorithm = algorithm.lower()

    # Each compiler expects specific neuron/synapse default parameters
    # (e.g. reset behaviour, current scaling)
    if algorithm == "eprop":
        from ml_genn.compilers.eprop_compiler import default_params
        synapse_factory = lambda: Delta()
    elif algorithm == "eventprop":
        from ml_genn.compilers.event_prop_compiler import default_params
        tau_syn = neuron_params.get("tau_syn", 5.0)
        synapse_factory = lambda: Exponential(tau=tau_syn)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    logging.info("Building SNN with configuration:")
    logging.info(f"  Input size:    {input_size}")
    logging.info(f"  Hidden layers: {hidden_layers}")
    logging.info(f"  Output size:   {output_size}")
    logging.info(f"  Readout:       {output_readout}")
    logging.info(f"  Algorithm:     {algorithm}")

    if max_input_spikes is None:
        max_input_spikes = 20000

    def hidden_weight_init(fan_in):
        if algorithm == "eventprop":
            # EventProp only propagates gradients through spikes, so the
            # hidden layer must fire from the start: use a positive weight
            # mean scaled so the summed input drive comfortably exceeds
            # threshold (cf. ml_genn's SHD/yin-yang examples)
            mean = 15.0 / fan_in
            return Normal(mean=mean, sd=0.5 * mean)
        return Normal(sd=1.0 / np.sqrt(fan_in))

    net = Network(default_params)
    with net:
        input_pop = Population(SpikeInput(max_spikes=max_input_spikes),
                               input_size, record_spikes=True)

        previous_layer = input_pop
        prev_units = input_size
        hidden_pops = []
        for layer_config in hidden_layers:
            hidden_units = layer_config["units"]
            hidden_neuron = LeakyIntegrateFire(
                v_thresh=neuron_params.get("threshold", 1.0),
                tau_mem=neuron_params.get("tau_mem", 20.0),
                tau_refrac=neuron_params.get("refractory_period", None))
            hidden_pop = Population(hidden_neuron, hidden_units,
                                    record_spikes=True)

            Connection(previous_layer, hidden_pop,
                       Dense(hidden_weight_init(prev_units)),
                       synapse_factory())

            if recurrent:
                # Recurrent hidden connectivity (RSNN) - both e-prop and
                # EventProp support this; init small so early dynamics are
                # dominated by the feed-forward drive
                if algorithm == "eventprop":
                    rec_init = Normal(mean=0.0, sd=0.02)
                else:
                    rec_init = Normal(sd=1.0 / np.sqrt(hidden_units))
                Connection(hidden_pop, hidden_pop, Dense(rec_init),
                           synapse_factory())

            hidden_pops.append(hidden_pop)
            previous_layer = hidden_pop
            prev_units = hidden_units

        # Output layer: a non-spiking leaky integrator whose membrane
        # voltage is read out as the prediction. Both e-prop and EventProp
        # train against this readout
        output_neuron = LeakyIntegrate(
            tau_mem=neuron_params.get("tau_mem", 20.0),
            readout=output_readout)
        output_pop = Population(output_neuron, output_size)

        # Connect every hidden layer directly to the output layer
        # (required by e-prop, harmless for EventProp)
        for hidden_pop in hidden_pops:
            if algorithm == "eventprop":
                out_init = Normal(mean=0.0, sd=0.03)
            else:
                out_init = Normal(sd=1.0 / np.sqrt(hidden_pop.shape[0]))
            Connection(hidden_pop, output_pop, Dense(out_init),
                       synapse_factory())

    return net, input_pop, output_pop


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    neuron_params = {
        "threshold": 1.0,
        "tau_mem": 20.0,
        "tau_syn": 5.0,
        "refractory_period": 1.0,
    }
    net, input_pop, output_pop = build_snn(
        input_size=50, hidden_layers=[{"units": 16}], output_size=1,
        neuron_params=neuron_params)
    logging.info(f"Built network with {len(net.populations)} populations "
                 f"and {len(net.connections)} connections")
