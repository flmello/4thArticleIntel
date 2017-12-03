//#include "stdafx.h"
//#include <intrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <string>


/*************** ADA_BOOST *******************/
#ifndef ADA_BOOST
#define ADA_BOOST 1
#include <vector>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>

namespace DM_AG{

  typedef std::vector<float> ClassificationResults;
  typedef boost::numeric::ublas::matrix<int> MatrixResults;
  typedef std::vector<int> Labels;

  // A classifier
  //    abstract class

  template <typename T>
  class Classifier {

  public:
    typedef typename boost::ptr_vector<Classifier<T> > CollectionClassifiers;
    typedef typename std::vector<T> Data;

    virtual int analyze(const T& feature) const = 0;
  };

  template <typename T>
  class ADA{

  public:

    //
    // Apply Adaboost
    //
    //  @param weak_classifiers, a set of weak classifiers
    //  @param data, the dataset to classify
    //  @param labels, classification labels (e.g. -1; +1}
    //  @param num_rounds, # boost iteration (default 100)

    ClassificationResults
    ada_boost(typename Classifier<T>::CollectionClassifiers const &weak_classifiers,
	      typename Classifier<T>::Data const & data,
	      const Labels & labels,
	      const unsigned int num_iterations){

      // following notation
      //
      // http://en.wikipedia.org/wiki/AdaBoost

      ClassificationResults alpha;
      ClassificationResults D;

      size_t labels_size = labels.size();
      size_t classifiers_size = weak_classifiers.size();

      D.resize(labels_size);           // D
      alpha.resize(labels_size);       // alpha

      // init result matrix for weak classifiers

      MatrixResults weak_classifiers_results(classifiers_size,
					     labels_size);
      // Run each weak classifer
      //
      unsigned int num_current_classifier = 0;
      typename Classifier<T>::CollectionClassifiers::const_iterator wc =
	weak_classifiers.begin();
      typename Classifier<T>::CollectionClassifiers::const_iterator wc_end =
	weak_classifiers.end();
      for (; wc!=wc_end; ++wc){

	/*std::cout << "\tClassifier=" << num_current_classifier << std::endl;*/

	for (unsigned int j=0; j < labels_size; j++){

	  // store the result for feature i
	  weak_classifiers_results(num_current_classifier, j) =
	    (*wc).analyze(data[j]);

	  //std::cout << " " << j << ","
		    weak_classifiers_results(num_current_classifier, j);
	}
	//std::cout << std::endl;
	num_current_classifier++;
      }

      // Init boosters
      for (unsigned int j=0; j < labels_size; j++)
	D[j] = (1.0) / labels_size;   // init D

      // for the maximum rounds
      //
      for (unsigned int round=0;
	   round < num_iterations; round++){

	//std::cout << "Iteration" << round << std::endl;

	float min_error=labels_size;
	unsigned int best_classifier = 0;

	//
	// for each classifier
	for (num_current_classifier = 0;
	     num_current_classifier < classifiers_size;
	     num_current_classifier++){

	  float error=0;

	  //
	  // for each feature
	  for (unsigned int j=0; j < labels_size; j++)
	    if (weak_classifiers_results(num_current_classifier,
					 j) != labels[j])
	      error += D[j];

	  if (error<min_error){
	    min_error = error; // this is the best observed
	    best_classifier = num_current_classifier;
	  }
	}// each classifier

	/*std::cout << "\tbest_classifier=" << best_classifier
		  << " error=" << min_error << std::endl;*/

	if (min_error >= 0.5)    // GOOD enough
	  break;                 // condition

	// a_t
	alpha[best_classifier] =
	  log((1.0f - min_error)/min_error)/2;

	// D_{t+1}
	ClassificationResults D_1(D);

	// update D_{t+1}
	float z = 0;
	for (unsigned int j=0; j < labels_size; j++){

	  D_1[j] *=
	    exp(-alpha[best_classifier] *
		labels[j] *
		weak_classifiers_results(best_classifier, j));
	  z+=D_1[j];
	}

	// normalize so that it is a prob distribution
	for (unsigned int j=0; j < labels_size; j++)
	  D[j] = D_1[j]/z;

      } // all the rounds.

      return alpha;
    };

  }; // class ADA

  //
  // A strong classifier is a linear comb of weak class
  //
  template <typename T>
  class StrongClassifier : public Classifier<T>
  {
  private:
    ClassificationResults & weigths_;
    size_t num_classifiers_;
    typename Classifier<T>::CollectionClassifiers * classifiers_;
    Labels & labels_;

  public:
    StrongClassifier(ClassificationResults & w,
		     typename Classifier<T>::CollectionClassifiers * cls,
		     Labels & labels) :
      weigths_(w),
      num_classifiers_(cls->size()),
      classifiers_(cls),
      labels_(labels){};

    // analyze
    //
    int analyze(const T & feature) const {

      float val=0;
      for (unsigned int current_classifier=0;
	   current_classifier < num_classifiers_;
	   current_classifier++)
	val += weigths_[current_classifier] *
	  (*classifiers_)[current_classifier].analyze(feature);

      if (val>=0)
	return 1;  // label +1
      else
	return -1; // label -1
    };

    void performance(){


    }


  }; // strong classifier

} // namespace

#endif





/*************** WEAK_CLASSIFIER *******************/
#ifndef WEAK_CLASSIFIER 
#define WEAK_CLASSIFIER 1

namespace DM_AG{

  class WeakClassifierOne : public Classifier<int>
  {
  public:
    int analyze(const int& i) const  {
      
      if (i>50 && i<90) return 1;
      return -1;
    }
  };
  
  class WeakClassifierTwo : public Classifier<int>
  {
  public:
    int analyze(const int& i) const {
      
      if (i>80 && i<130) return 1;
      return -1;
    }
  };
  
  class WeakClassifierThree : public Classifier<int>
  {
  public:
    int analyze(const int& i) const  {
      
      if (i>90 && i<130) return 1;
      return -1;
    }
  };
  
  class WeakClassifierFour : public Classifier<int>
  {
  public:
    int analyze(const int& i) const {
      
      if (i>40 && i<150) return 1;
      return -1;
    }
  };
};
#endif




using namespace DM_AG;

uint64_t rdtsc() {
	return __rdtsc();
}


/*************** MAIN *******************/
int main(){

  Classifier<int>::Data data;
  Labels labels;
  std::ifstream file("data/adaboost_data_train.csv");
  std::string value;
  double value2;
 
  uint64_t uiInicio, uiFim;
  int iTam = 1000;

  const unsigned int number_features = 20;

  // SIMULATE A TRAINER

  //std::cout << "Training ... " << std::endl;

  for (unsigned int i=0; i < number_features; i++){	  
  	getline(file, value, ',');
  	value2 = stod(value);
    data.push_back(value2);
    // ground-truth classifier (TRAINER)
      
    (i>7&&i<13) ? 
      labels.push_back(1) :  // good
      labels.push_back(-1);  // bad
  }
  //std::cout << "Done ... " << std::endl;
  //std::cout << "WeakClassifier ... " << std::endl;
  //
  // Create the pool of classifiers
  //
  Classifier<int>::CollectionClassifiers classifiers;
  classifiers.push_back(new WeakClassifierOne());
  classifiers.push_back(new WeakClassifierTwo());
  classifiers.push_back(new WeakClassifierThree());
  classifiers.push_back(new WeakClassifierFour());

  //
  // Ada boosting
  //
  ADA<int> ada;
  //std::cout << "Boosting ... " << std::endl;

  uiInicio = rdtsc();
  for (int i = 0; i < iTam; i++) {
    ClassificationResults weights = ada.ada_boost(classifiers, data, labels, 100);

    unsigned int classifiers_size = classifiers.size();
     
    //Start the strong classifier
    
    StrongClassifier<int> sc(weights, &classifiers, labels);

    for (unsigned int i=0; i < number_features; i++){
      sc.analyze(i);
    }

    sc.performance();
  }
  uiFim = rdtsc();
  
  std::cout << (uiFim - uiInicio) / iTam;
 
  return 0;
}
