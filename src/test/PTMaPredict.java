package test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import topic.PTMa;
import util.Corpus;
import util.EvaluationPTM;
import util.ReadWriteFile;

public class PTMaPredict {

	public static void main(String[] args) throws IOException {

		StringBuilder sb = new StringBuilder();

		int K = 20;

		int N = 5;

		for (int i = 0; i < 10; i++) {

			List<String> herbs_list = Corpus.getVocab("D:\\PTM\\PTM\\data\\LFLDA\\herb_embedding.txt");

			List<String> symptoms_list = Corpus.getVocab("D:\\PTM\\PTM\\data\\LFLDA\\symp_embedding.txt");

			int[][] herbs_train = Corpus.getDocuments("D:\\PTM\\PTM\\data\\LFLDA\\herbs_train.txt");

			int[][] symptoms_train = Corpus.getDocuments("D:\\PTM\\PTM\\data\\LFLDA\\symps_train.txt");

			int[][] herbs_test = Corpus.getDocuments("D:\\PTM\\PTM\\data\\LFLDA\\herbs_test.txt");

			int[][] symptoms_test = Corpus.getDocuments("D:\\PTM\\PTM\\data\\LFLDA\\symps_test.txt");

			PTMa ptm = new PTMa(herbs_train, symptoms_train, herbs_list.size(), symptoms_list.size());

			double alpha = 1;
			double beta = 0.1;
			double beta_bar = 0.1;
			double eta = 1;
			int iterations = 1000;

			ptm.markovChain(K, alpha, beta, beta_bar, eta, iterations);

			double[][][] herb_topic = ptm.estimatePhi();

			double[][] symptom_topic = ptm.estimatePhiBar();

			double[][][] prescription_topic_role = ptm.estimatePi();

			double[][] prescription_topic = ptm.estimateTheta();

			BufferedWriter writer = new BufferedWriter(new FileWriter("C:\\Users\\wxy\\Desktop\\PTM.phiHerb"));
			for (int t = 0; t < K; t++) {
				for (int x = 0; x < 4; x++) {
					for(int h = 0; h < herb_topic[0][0].length; h++){
						writer.write(herb_topic[t][x][h] + " ");
					}
				}
				writer.write("\n");
			}
			writer.close();

			BufferedWriter writerS = new BufferedWriter(new FileWriter("C:\\Users\\wxy\\Desktop\\PTM.phiSymp"));
			for (int t = 0; t < K; t++) {
				for(int s = 0; s < symptom_topic[0].length; s++){
					writerS.write(symptom_topic[t][s]+ " ");

				}
				writerS.write("\n");
			}
			writerS.close();

//			for(int k = 0; k < K; k++){
//				for(int s = 0; s < symptoms_list.size(); s++){
//					System.out.println("症状"+s+"\t主题"+k+"\t:"+symptom_topic[k][s]);
//				}
//			}



//			double symptom_perplexity = EvaluationPTM.ptm_symptom_predictive_perplexity(herbs_test, symptoms_test,
//					herb_topic, symptom_topic);
//
//			System.out.println("PTM(a) symptom predictive perplexity : " + symptom_perplexity);
//
//			double symptom_precision_k = EvaluationPTM.ptm_symptom_precision_k(herbs_test, symptoms_test, herb_topic,
//					symptom_topic, N);
//
//			System.out.println("PTM(a) symptom precision@" + N + ": " + symptom_precision_k);
//
//			double symptom_recall_k = EvaluationPTM.ptm_symptom_recall_k(herbs_test, symptoms_test, herb_topic,
//					symptom_topic, N);
//
//			System.out.println("PTM(a) symptom recall@" + N + ": " + symptom_recall_k);
//
//			double symptom_ndcg_k = EvaluationPTM.ptm_symptom_ndcg(herbs_test, symptoms_test, herb_topic, symptom_topic,
//					N);
//
//			System.out.println("PTM(a) symptom NDCG@" + N + ": " + symptom_ndcg_k);

			double herb_perplexity = EvaluationPTM.ptm_herb_predictive_perplexity(herbs_test, symptoms_test, herb_topic,
					symptom_topic, prescription_topic_role, prescription_topic);

			System.out.println("PTM(a) herb predictive perplexity : " + herb_perplexity);

			double herb_precision_k = EvaluationPTM.ptm_herb_precision_k(herbs_test, symptoms_test, herb_topic,
					symptom_topic, prescription_topic_role, prescription_topic, N);

			System.out.println("PTM(a) herb precision@" + N + ": " + herb_precision_k);

			double herb_recall_k = EvaluationPTM.ptm_herb_recall_k(herbs_test, symptoms_test, herb_topic, symptom_topic,
					prescription_topic_role, prescription_topic, N);

			System.out.println("PTM(a) herb recall@" + N + ": " + herb_recall_k);

			double herb_ndcg_k = EvaluationPTM.ptm_herb_ndcg(herbs_test, symptoms_test, herb_topic, symptom_topic,
					prescription_topic_role, prescription_topic, N);

			System.out.println("PTM(a) herb NDCG@" + N + ": " + herb_ndcg_k);
//			sb.append(herb_perplexity + "," + herb_precision_k + "," + symptom_perplexity + "," + symptom_precision_k
//					+ "\n");

		}

		ReadWriteFile.writeFile("file//ptm_a_" + K + "_" + N + ".csv", sb.toString());

	}

}