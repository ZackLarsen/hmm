public class ReadFile {
    List<List<String>> records = new ArrayList<>();
    try (BufferedReader br = new BufferedReader(new FileReader("../WSJ_head.csv"))) {
        String line;
        while ((line = br.readLine()) != null) {
            String[] values = line.split(COMMA_DELIMITER);
            records.add(Arrays.asList(values));
        }
    }
}
