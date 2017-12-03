#include "daal.h"
#include "service.h"

//#include <intrin.h>
#include <stdint.h>

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string trainDatasetFileName     = "data/adaboost_data_train.csv";

string testDatasetFileName      = "data/adaboost_data_test.csv";

const size_t nFeatures = 20;

services::SharedPtr<adaboost::training::Result> trainingResult;
services::SharedPtr<classifier::prediction::Result> predictionResult;
NumericTablePtr testGroundTruth;

void trainModel();
void testModel();
void printResults();

uint64_t rdtsc() {
	return __rdtsc();
}


int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and labels */
    NumericTablePtr trainData(new HomogenNumericTable<double>(nFeatures, 0, NumericTable::notAllocate));
    NumericTablePtr trainGroundTruth(new HomogenNumericTable<double>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainGroundTruth));

    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to train the AdaBoost model */
    adaboost::training::Batch<> algorithm;

    /* Pass the training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);    
	
    //Inicio da medicao de tempo
    uint64_t uiInicio, uiFim;
    int iTam = 1000;

	uiInicio = rdtsc();
	for (int i = 0; i < iTam; i++) {
        /* Train the AdaBoost model */
        algorithm.compute();

        /* Retrieve the results of the training algorithm  */
        trainingResult = algorithm.getResult();
	}
	uiFim = rdtsc();

    testModel();
    //printResults();

	//Fim da medicao de tempo
	

	cout << (uiFim - uiInicio)/iTam;

    return 0;
}

void testModel()
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::notAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData(new HomogenNumericTable<double>(nFeatures, 0, NumericTable::notAllocate));
    testGroundTruth = NumericTablePtr(new HomogenNumericTable<double>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create algorithm objects for AdaBoost prediction with the default method */
    adaboost::prediction::Batch<> algorithm;

    /* Pass the testing data set and trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data,  testData);
    algorithm.input.set(classifier::prediction::model, trainingResult->get(classifier::training::model));

    /* Compute prediction results */
    algorithm.compute();

    /* Retrieve algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults()
{
    printNumericTables<int, int>(testGroundTruth,
                                 predictionResult->get(classifier::prediction::prediction),
                                 "Ground truth", "Classification results",
                                 "AdaBoost classification results (first 20 observations):", 20);
}
