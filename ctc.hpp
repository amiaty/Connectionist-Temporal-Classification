/*
	Original Author:
			Alex Graves (2009-2010)
			
	Modified and refactored by
			Amir Ahooye Atashin (2015-2016)
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include "dataset.h"
#include "Log.hpp"

using namespace std;

// ref: http://stackoverflow.com/questions/9694838/how-to-implement-2d-vector-array

#define vector_double vector<double>
#define matrix_double vector<vector_double>
#define LogD Log<double>
#define vector_logd vector<LogD>
#define matrix_logd vector<vector_logd>


const Log<double> logOne1(1);

struct InputDataCTC
{
    //data
    vector<int> targetLabelSeq;
    matrix_logd inputs;
    matrix_double outputErrors;
	//matrix_double outputErrorsE;
    int blank_id;
    void LoadDataFrom_HCRF_Toolbox(DataSequence &ds, dMatrix* probabilities)
    {
        for(int i = 0; i < ds.getStateLabels()->getLength(); ++i)
        {
            targetLabelSeq.push_back(ds.getStateLabels(i));
        }
        int row = probabilities->getWidth();
        int col = probabilities->getHeight();
        blank_id = col - 1;
        inputs.resize(row, vector_logd(col, LogD(0)));
        outputErrors.resize(row, vector_double(col, 0));
		//outputErrorsE.resize(row - 1, vector_double((col - 1) * (col - 1), 0));
        for(int i = 0; i < row; ++i)
            for (int j = 0; j < col; ++j) {
				double p = 0;
				//if (j != blank_id)
				p = probabilities->getValue(j, i);
                //cerr << p << endl;
                LogD t(p);
                inputs[i][j] = t;
            }
    }
    /*
    void LoadDataFrom_RNN_Lib(const DataSequence &ds, SeqBuffer<LogD> actv)
    {
        int row = actv.seq_size();
        int col = actv.depth;
        //cerr << row << endl;
        inputs.resize(row, vector_logd(col, LogD(0)));
        outputErrors.resize(row, vector_double(col, 0));
        for(int i = 0; i < row; ++i)
            for (int j = 0; j < col; ++j) {
                inputs[i][j] = actv[i][j];
            }
        targetLabelSeq = ds.targetLabelSeq;
    }
    */
};
class CTC
{
    matrix_logd forwardVariables;
    matrix_logd backwardVariables;
    int blank;
    int totalSegments;
    int totalTime;
    vector_logd dEdYTerms;
    vector<int> outputLabelSeq;
public:
    CTC()
    {
        blank = -1;
    }
    CTC(int blank_ind)
    {
        blank = blank_ind;
    }
    double calculate_errors(InputDataCTC &seq, bool fullEx)
    {
        if(blank == -1)
            blank = seq.blank_id;
        totalTime = (int)seq.inputs.size();
        int outSize = (int)seq.inputs[0].size();

        int len = (int)seq.targetLabelSeq.size();
		/*
        int requiredTime = len;
        int oldLabel = -1;
        for (int i = 0; i < len; ++i)
        {
            if (seq.targetLabelSeq[i] == oldLabel)
            {
                ++requiredTime;
            }
            oldLabel = seq.targetLabelSeq[i];
        }
		
        if (totalTime < requiredTime)
        {
            std::cerr << "Error: seq data has requiredTime " << requiredTime << " > totalTime " << totalTime << endl;
            return -1.0;
        }
		*/
        totalSegments = ((int)seq.targetLabelSeq.size() * 2) + 1;

        /*	calculate the forward variables*/
        forwardVariables.resize(totalTime, vector_logd(totalSegments, LogD(0)));

        forwardVariables[0][0] = seq.inputs[0][blank];
        //cerr << forwardVariables[0][0] << endl;
        if (totalSegments > 1)
        {
            forwardVariables[0][1] = seq.inputs[0][seq.targetLabelSeq[0]];
            //cerr << forwardVariables[0][1] << endl;
        }

        for (int t = 1; t < totalTime; ++t)
        {
            vector_logd logActs = seq.inputs[t];
            vector_logd oldFvars = forwardVariables[t - 1];
            vector_logd fvars = forwardVariables[t];
            vector<int> srange = segment_range(t);
            int s;
            for(int i = 0; i < srange.size(); ++i)
            {
                s = srange[i];
                LogD fv(0);
                //if(t == 2) cerr << "S:" << s << endl;
                //s odd (label output)
                if (s & 1)
                {
                    int labelIndex = s / 2;
                    int labelNum = seq.targetLabelSeq[labelIndex];
					fv += oldFvars[s];
					fv += oldFvars[s - 1];
                    //if(s == 1) cerr << "G:" << fv.log() << endl;
                    if (s > 1)// && labelNum != seq.targetLabelSeq[labelIndex - 1])
                    {
                        fv += oldFvars[s - 2];
                    }
					fv *= (logActs[labelNum] *prior_label_prob(labelIndex));
                    //if(s == 1) cerr << "G:" << fv.log() << endl;
                }
                //s even (blank output)
                else
                {
                    fv = oldFvars[s];
                    if (s)
                    {
                        fv += oldFvars[s - 1];
                    }
                    fv *= logActs[blank];
                }
                fvars[s] = fv;
            }
            forwardVariables[t] = fvars;
        }
        vector_logd lastFvs = forwardVariables[totalTime - 1];
        LogD logProb = lastFvs.back();
        if (totalSegments > 1)
        {
            logProb += lastFvs[lastFvs.size()-2];
        }
		double ctcError = -logProb.log();
		if (!fullEx)
			return ctcError;
        //cerr << "d " << logProb.log() << endl;
        //calculate the backward variables
        backwardVariables.resize(totalTime, vector_logd(totalSegments, LogD(0)));
        vector_logd lastBvs = backwardVariables[totalTime - 1];
        lastBvs.back() = LogD(1);
        if (totalSegments > 1)
        {
            lastBvs[lastBvs.size()-2] = LogD(1);
        }
        backwardVariables[totalTime - 1] = lastBvs;
        //LOOP over time, calculating backward variables recursively
        for (int t = totalTime - 2; t >= 0; --t)
        {
            vector_logd oldLogActs = seq.inputs[t + 1];
            vector_logd oldBvars = backwardVariables[t + 1];
            vector_logd bvars = backwardVariables[t];
            vector<int> srange = segment_range(t);
            int s;
            for(int i = 0; i < srange.size(); ++i)
            {
                LogD bv(0);
                s = srange[i];
                //s odd (label output)
                if (s & 1)
                {
                    int labelIndex = s / 2;
                    int labelNum = seq.targetLabelSeq[labelIndex];
					bv = (oldBvars[s] * oldLogActs[labelNum] *prior_label_prob(labelIndex)) + (oldBvars[s + 1] * oldLogActs[blank]);
                    if (s < (totalSegments - 2))
                    {
                        int nextLabelNum = seq.targetLabelSeq[labelIndex + 1];
                        //if (labelNum != nextLabelNum)
                        {
							bv += (oldBvars[s + 2] * oldLogActs[nextLabelNum] *prior_label_prob(labelIndex + 1));
                        }
                    }
                    //if(s == 1) cerr << "B:" << bv.log() << endl;
                }

                //s even (blank output)
                else
                {
                    bv = oldBvars[s] * oldLogActs[blank];

                    if (s < (totalSegments - 1))
                    {
						bv += (oldBvars[s + 1] * oldLogActs[seq.targetLabelSeq[s / 2]] *prior_label_prob(s / 2));
                    }
                }
                bvars[s] = bv;
            }
            backwardVariables[t] = bvars;
        }
		
        for(int time = 0; time < totalTime; ++time)
        {
            vector_logd fvars = forwardVariables[time];
            vector_logd bvars = backwardVariables[time];
            dEdYTerms.resize(outSize, LogD(0));
            for (int s = 0; s < totalSegments; s++)
            {
                //k = blank for even s, target label for odd s
                int k = (s & 1) ? seq.targetLabelSeq[s / 2] : blank;
                dEdYTerms[k] += (fvars[s] * bvars[s]);
            }
            //if(time == 2) cerr << "safasf" << endl;
			
            for (int j = 0; j < outSize; j++)
            {
                seq.outputErrors[time][j] = (dEdYTerms[j] / (logProb * seq.inputs[time][j])).exp();
            }
			/*
			vector_logd dEdYTerms_prev;
			if (time != 0)
			{
				for (int j = 0; j < outSize - 1; j++)
				{
					for (int k = 0; k < outSize - 1; k++)
					{
						seq.outputErrorsE[time - 1][j* (outSize - 1) + k] = ( (dEdYTerms_prev[j] / seq.inputs[time - 1][j] + dEdYTerms[k]/ seq.inputs[time][k]) / logProb ).exp();
					}
				}
			}
			dEdYTerms_prev = dEdYTerms;
			*/
            dEdYTerms.clear();
        }

        return ctcError;
    }
    virtual const LogD& prior_label_prob(int label)
    {
        return logOne1;
    }
    vector<int> segment_range(int time, int totalSegs = -1) const
    {
        if (totalSegs < 0)
        {
            totalSegs = totalSegments;
        }

        int start = (int)fmax(0, totalSegs - (2 * (totalTime - time)));
        int end = (int)fmin(totalSegs, 2 * (time + 1));

        vector<int> range(end - start);
        int k = 0;
        for (int i = start; i < end; i++)
            range[k++] = i;

        return range;
    }
};

