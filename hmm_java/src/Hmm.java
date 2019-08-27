/**
 * This is a class to train hidden markov models (HMM)
 * version: 1.0 August 26, 2019
 * author: Zack Larsen
 */

import java.util.*;

public class Hmm {

    public static void main(String[] args)
    {

        int n_states = Integer.parseInt(args[0]);
        int n_observations = Integer.parseInt(args[1]);
        System.out.println("The dimensions of this array are " + n_states +  " by " + n_observations);

        double[][] transitions = {
                {0.3777, 0.0110, 0.0009, 0.0084, 0.0584, 0.0090, 0.0025},
                {0.0008, 0.0002, 0.7968, 0.0005, 0.0008, 0.1698, 0.0041},
                {0.0322, 0.0005, 0.0050, 0.0837, 0.0615, 0.0514, 0.2231},
                {0.0366, 0.0004, 0.0001, 0.0733, 0.4509, 0.0036, 0.0036},
                {0.0096, 0.0176, 0.0014, 0.0086, 0.1216, 0.0177, 0.0068},
                {0.0068, 0.0102, 0.1011, 0.1012, 0.0120, 0.0728, 0.0479},
                {0.1147, 0.0021, 0.0002, 0.2157, 0.4744, 0.0102, 0.0017}
        };

        double[][] emissions = {
                {0.000032, 0, 0, 0.000048, 0},
                {0, 0.308431, 0, 0, 0},
                {0, 0.000028, 0.000672, 0, 0.000028},
                {0, 0, 0.000340, 0, 0},
                {0, 0.000200, 0.000223, 0, 0.002337},
                {0, 0, 0.010446, 0, 0},
                {0, 0, 0, 0.506099, 0}
        };

        double[] Pi = {0.2767, 0.0006, 0.0031, 0.0453, 0.0449, 0.0510, 0.2026};

        System.out.println("The dimensions of the transitions array are " + transitions.length + " by " + transitions[0].length);
        System.out.println("The dimensions of the emissions array are " + emissions.length + " by " + emissions[0].length);
        System.out.println("The dimensions of the Pi array are " + "1" + " by " + Pi.length);

        var integer_observations = new HashMap<Int, String>();
        integer_observations.put(0, new String("Janet"));
        integer_observations.put(1, new String("will"));
        integer_observations.put(2, new String("back"));
        integer_observations.put(3, new String("the"));
        integer_observations.put(4, new String("bill"));

        System.out.println(integer_observations);

        // Lookup a value
        System.out.println(integer_observations.get("Janet"));

        // Iterate through all key, value pairs:
        integer_observations.forEach((k, v) -> System.out.println("key=" + k + ", value=" + v));

    }

}




/**
 * Here is the original data from the SLP textbook, chapter 8 (POS Tagging)
 *
 *
<s>0.2767 0.0006 0.0031   0.0453  0.0449   0.0510   0.2026

NNP MD VB JJ NN RB DT
NNP 0.37770.01100.0009   0.0084  0.0584   0.0090   0.0025
MD 0.00080.00020.7968   0.0005  0.0008   0.1698   0.0041
VB 0.03220.00050.0050   0.0837  0.0615   0.0514   0.2231
JJ 0.03660.00040.0001   0.0733  0.4509   0.0036   0.0036
NN 0.00960.01760.0014   0.0086  0.1216   0.0177   0.0068
RB 0.00680.01020.1011   0.1012  0.0120   0.0728   0.0479
DT 0.1147 0.0021 0.0002   0.2157  0.4744   0.0102   0.0017

Janet will back the bill
NNP 0.000032  0 0 0.000048  0
MD 0 0.308431  0 0 0
VB 0 0.000028  0.000672  0 0.000028
JJ 0 0 0.00034 0 0 0
NN 0 0.000200 0.000223 0 0.002337
RB 0 0 0.010446 0 0
DT 0 0 0 0.506099 0
 */