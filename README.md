# Straggler getting delta model updates

The experiment should be run with `test_hello_federation.sh`

The following happens:
1. Three cols are created,
2. Only 2 start, training goes for 2 rounds
3. The second collaborator becomes a straggler (sleeps)
4. The third collaborator connects and training goes on

To learn the exact sleeping strategy for collaborators see train method in src/keras_cnn.py KerasCNN class.
Straggler handling plugin is turned on: the aggregator waits for 2 cols out of 3.

Result of the experiment: training goes as expected, delta updates are handled properly for a newly connected collagorator, see https://github.com/intel/openfl/blob/b0099cdabc96aa2d20ab3f4564d85792d5bf3781/openfl/component/collaborator/collaborator.py#L322
