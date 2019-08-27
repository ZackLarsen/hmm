public class FileReader {

    public static void main(String[] args) throws IOException
    {
        // Here is where we read in the file
        Scanner in = new Scanner(Path.of(args[0]), StandardCharsets.UTF_8);
    }

}
