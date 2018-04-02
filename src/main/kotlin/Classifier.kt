import weka.classifiers.AbstractClassifier
import weka.classifiers.Evaluation
import weka.core.Instances
import weka.core.converters.CSVLoader
import java.io.File
import java.util.*

object Classifier {
    fun loadData(): Instances {
        val filePath = this.javaClass.getResource("/lenses.csv").toURI().path
        val file = File(filePath)

        val csvLoader = CSVLoader()
        csvLoader.noHeaderRowPresent = true
        csvLoader.fieldSeparator = ";"
        csvLoader.setSource(file)
        csvLoader.nominalAttributes = "1-5"  // All columns must are nominal.

        val data = csvLoader.dataSet

        val unknownClassIndex = data.classIndex() == -1

        if (unknownClassIndex) {
            println("Setting class index.")
            data.setClassIndex(data.numAttributes() - 1)
        }

        return data
    }

    fun evaluateModel(model: AbstractClassifier, data: Instances, folds: Int = 10, seed: Long = 42) {
        val evaluator = Evaluation(data)

        evaluator.crossValidateModel(model, data, folds, Random(seed))
        val summary = evaluator.toSummaryString("Results", false)

        println(summary)
        println("Estimated accuracy after $folds folds: ${evaluator.pctCorrect()}")
    }
}
