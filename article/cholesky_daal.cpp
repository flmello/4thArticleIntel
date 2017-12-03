#include "daal.h"
#include "service.h"
//#include <intrin.h>
#include <stdint.h>

using namespace std;
using namespace daal;
using namespace daal::algorithms;

uint64_t rdtsc() {
	return __rdtsc();
}

/* Input data set parameters */
string datasetFileName = "data/cholesky_data.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

	//Inicio da medicao de tempo
	uint64_t uiInicio, uiFim;
	int iTam = 1000;

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm to compute Cholesky decomposition using the default method */
    cholesky::Batch<> algorithm;

    /* Set input objects for the algorithm */
    algorithm.input.set(cholesky::data, dataSource.getNumericTable());

	uiInicio = rdtsc();
	for (int i = 0; i < iTam; i++) {
		/* Compute Cholesky decomposition */
		algorithm.compute();
	}
	uiFim = rdtsc();
    /* Get computed Cholesky decomposition */
    services::SharedPtr<cholesky::Result> res = algorithm.getResult();

    //printNumericTable(res->get(cholesky::choleskyFactor));

	cout << (uiFim - uiInicio) / iTam << endl;
    return 0;
}
