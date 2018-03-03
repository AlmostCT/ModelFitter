package top.almostct.foodhack.model

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import java.io.BufferedReader
import java.io.FileReader
import java.io.InputStream
import java.io.InputStreamReader
import java.util.regex.Pattern
import kotlin.streams.toList

class DatasetCreator(private val inputStream : InputStream) {

    val map = hashMapOf<String, Int>()

    fun transform(sentenceSize : Int) : DataSetIterator {
        val result = BufferedReader(InputStreamReader(inputStream))
                .lines().map { t -> t.split(',') }.toList()

        val data = result.map { t -> t[0]}.toList()
        val labels = result.map { t -> t[1].toInt() }.toList()

        var index = 0;
        for (s in data) {
            for (word in s.split(Pattern.compile("\\s"))) {
                if (word.equals("")) {
                    throw RuntimeException("Illegal empty string")
                }
                if (!map.containsKey(word)) {
                    map[word] = index++;
                }
            }
        }


        val dataset = ArrayList<IntArray>()
        for (s in data) {
            val features = IntArray(sentenceSize)
            val words = s.split(Pattern.compile("\\s"))
            for (i in 0 until sentenceSize) {
                if (i < words.size)
                    features[i] = map[words[i]]!!
                else
                    features[i] = map.size
            }
            dataset.add(features)
        }
        return CommandsDatasetIterator(dataset, labels, map.size + 1)
    }


    class CommandsDatasetIterator(private val rawFeutures : List<IntArray>,
                                  private val rawLabels :  List<Int>,
                                  private val inputSize: Int) : DataSetIterator {

        private val labelesCount = rawLabels.max()!! + 1
        private var cursor : Int = 0

        override fun getLabels(): MutableList<String>? {
            return null
        }

        override fun cursor(): Int {
            return cursor
        }

        override fun remove() {
            throw UnsupportedOperationException()
        }

        override fun inputColumns(): Int {
            return labelesCount
        }

        override fun numExamples(): Int {
            return rawFeutures.size
        }

        override fun batch(): Int {
            return rawFeutures.size
        }

        override fun next(num: Int): DataSet {

            if (!hasNext()) {
                throw IllegalStateException("There is no such element")
            }

            var timeSeriesLength = rawFeutures[0].size
            var actualExamples = rawFeutures.size

            val features = Nd4j.create(intArrayOf(actualExamples, inputSize, timeSeriesLength),'f')
            val labels = Nd4j.zeros(intArrayOf(actualExamples, labelesCount, timeSeriesLength), 'f')

            for (i in 0..actualExamples - 1) {
                val actualData = rawFeutures[i]
                for (j in 0..timeSeriesLength - 1) {
                    var scalar = actualData[j]
                    features.putScalar(intArrayOf(i, scalar, j), 1.0);
                }
            }

            for (i in 0..actualExamples - 1) {
                var label = rawLabels[i]
                labels.putScalar(intArrayOf(i, label, timeSeriesLength - 1), 1f)
            }

            val labelsMask = Nd4j.zeros(intArrayOf(actualExamples, timeSeriesLength), 'f')
            for (i in 0..actualExamples - 1) {
                labelsMask.putScalar(intArrayOf(i, timeSeriesLength - 1), 1f)
            }

            /* println("----- labels -----")
             for (kvp in labelsCount) {
                 println("${kvp.key} : ${kvp.value}")
             }*/
            cursor = 1
            return DataSet(features, labels, null, labelsMask)
        }

        override fun next(): DataSet {
            return next(batch())
        }

        override fun totalOutcomes(): Int {
            return labelesCount
        }

        override fun setPreProcessor(preProcessor: DataSetPreProcessor?) {
            throw UnsupportedOperationException()
        }

        override fun totalExamples(): Int {
            return rawFeutures.size
        }

        override fun reset() {
            cursor = 0
        }

        override fun hasNext(): Boolean {
            return cursor == 0
        }

        override fun asyncSupported(): Boolean {
            return false
        }

        override fun getPreProcessor(): DataSetPreProcessor? {
            return null
        }

        override fun resetSupported(): Boolean {
            return true
        }

    }
}