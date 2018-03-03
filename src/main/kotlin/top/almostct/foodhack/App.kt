import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.WorkspaceMode
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT
import top.almostct.foodhack.model.DatasetCreator
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import java.util.regex.Pattern

fun main(args : Array<String>) {
    Thread.currentThread().contextClassLoader.getResourceAsStream("rawDataset.csv").use { inputStream ->
        val sentenceSize = 10
        val creator = DatasetCreator(inputStream)
        val datasetIterator = creator.transform(sentenceSize)
        println(creator.map.size)
        val dataset = datasetIterator;

        var model = getModel(creator.map.size + 1, 4)
        val outputPath = Paths.get("Models")

        if (Files.exists(outputPath.resolve("CommandsRecognition.zip"))) {
           model =  ModelSerializer.restoreComputationGraph(outputPath.resolve("CommandsRecognition.zip").toAbsolutePath().toString())
        } else {
            for (i in 0..1000) {
                model.fit(datasetIterator)
                println("epoch end")
                datasetIterator.reset()
            }
            if (!Files.exists(outputPath)) {
                Files.createDirectories(outputPath)
            }
            ModelSerializer.writeModel(model, outputPath.resolve("CommandsRecognition.zip").toString(), true)

        }

        val array = sentenceToDataset(sentenceSize, "А дальше то что следует сделать", creator.map)
        val output = model.rnnTimeStep(array)
        val lastTimeStep = output[0].tensorAlongDimension(sentenceSize - 1,1,0)

        val classesCount = 4
        var curIdx = 0
        var curValue = lastTimeStep.getDouble(0)

        for (i in 0 until classesCount) {
            if (curValue < lastTimeStep.getDouble(i)) {
                curIdx = i
                curValue = lastTimeStep.getDouble(i)
            }
        }
        println(curIdx)
    }
}

fun sentenceToDataset(sentenceSize : Int, sentence : String, map : HashMap<String, Int>) : INDArray {
    val data = IntArray(sentenceSize, {i -> map.size})
    val words = sentence.split(Pattern.compile("\\s"))
    for (i in 0 until sentenceSize) {
        if (i < words.size) {
            if (map.containsKey(words[i])) {
                data[i] = map[words[i]]!!;
            }
        }
    }

    val array = Nd4j.zeros(1, map.size + 1, sentenceSize)
    for (i in 0 until sentenceSize) {
        array.putScalar(intArrayOf(0, data[i], i), 1.0)
    }
    return array
}

fun getModel(inputCount : Int, outputCount : Int) : ComputationGraph {
    val innerNeuronCount = 64

    val config = NeuralNetConfiguration.Builder()
            .trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.RELU)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .updater(Updater.ADAM) //NESTEROVS
            .learningRate(0.005)
            .graphBuilder()
            .addInputs("input")
            .addLayer("LSTM-1",  GravesLSTM.Builder().activation(Activation.TANH)
                    .nIn(inputCount).nOut(innerNeuronCount).dropOut(0.2).build(), "input")
            .addLayer("Output", RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .activation(Activation.SOFTMAX).nIn(innerNeuronCount).nOut(outputCount).build() ,"LSTM-1")
            .setInputTypes(InputType.recurrent(inputCount))
            .setOutputs("Output")
            .pretrain(false).backprop(true).build()

    val net = ComputationGraph(config)
    net.init()
    net.addListeners(ScoreIterationListener(1))
    return net
}