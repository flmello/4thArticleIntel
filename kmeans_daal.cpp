//Remenber: source /opt/intel/compilers_and_libraries/linux/daal/bin/daalvars.sh intel64

#include "daal.h"
#include "service.h"
//#include <intrin.h>
#include <stdint.h>


using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string datasetFileName     = "data/kmeans_data.csv";

/* K-Means algorithm parameters */
const size_t nClusters   = 20;
const size_t nIterations = 5;

uint64_t rdtsc() {
	return __rdtsc();
}

int main(int argc, char *argv[])
{
	uint64_t uiInicio, uiFim;
	int iTam = 1000;

    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

	uiInicio = rdtsc();

	for (int i = 0; i < iTam; i++) {
		/* Get initial clusters for the K-Means algorithm */
		kmeans::init::Batch<double, kmeans::init::randomDense> init(nClusters);

		init.input.set(kmeans::init::data, dataSource.getNumericTable());
		init.compute();

		NumericTablePtr centroids = init.getResult()->get(kmeans::init::centroids);

		
		/* Create an algorithm object for the K-Means algorithm */
		kmeans::Batch<> algorithm(nClusters, nIterations);

		algorithm.input.set(kmeans::data, dataSource.getNumericTable());
		algorithm.input.set(kmeans::inputCentroids, centroids);

		algorithm.compute();
		}

	//Fim da medicao de tempo
	uiFim = rdtsc();

	cout << (uiFim - uiInicio) / iTam << endl;

    ///* Print the clusterization results */
    //printNumericTable(algorithm.getResult()->get(kmeans::assignments), "First 10 cluster assignments:", 10);
    //printNumericTable(algorithm.getResult()->get(kmeans::centroids  ), "First 10 dimensions of centroids:", 20, 10);
    //printNumericTable(algorithm.getResult()->get(kmeans::goalFunction), "Goal function value:");

    return 0;
}
