import Classifier.evaluateModel
import Classifier.loadData
import weka.classifiers.trees.J48

fun main(args: Array<String>) {
    val data = loadData()
    evaluateModel(J48(), data)
}