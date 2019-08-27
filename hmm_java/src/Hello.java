/**
 * This is a class to say hello!
 * version: 1.0 August 26, 2019
 * author: Zack Larsen
 */

public class Hello
{
    public static void main(String[] args)
    {
        if (args[0].equals("g"))
            System.out.print("Goodbye,");
        else if (args[0].equals("h"))
            System.out.print("Hello,");
        // Printing other arguments
        for (int i = 1; i <args.length; i++)
            System.out.print(" " + args[i]);
        System.out.println(" world!");
    }
}
