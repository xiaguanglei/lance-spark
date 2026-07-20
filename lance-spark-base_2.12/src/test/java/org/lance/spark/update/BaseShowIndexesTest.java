/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance.spark.update;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/** Base test for SHOW INDEXES command. */
public abstract class BaseShowIndexesTest {
  protected String catalogName = "lance_test";
  protected String tableName = "show_indexes_test";
  protected String fullTable = catalogName + ".default." + tableName;

  protected SparkSession spark;

  @TempDir Path tempDir;
  protected String tableDir;

  @BeforeEach
  public void setup() throws IOException {
    Path rootPath = tempDir.resolve(UUID.randomUUID().toString());
    Files.createDirectories(rootPath);
    String testRoot = rootPath.toString();
    spark =
        SparkSession.builder()
            .appName("lance-show-indexes-test")
            .master("local[10]")
            .config(
                "spark.sql.catalog." + catalogName, "org.lance.spark.LanceNamespaceSparkCatalog")
            .config(
                "spark.sql.extensions", "org.lance.spark.extensions.LanceSparkSessionExtensions")
            .config("spark.sql.catalog." + catalogName + ".impl", "dir")
            .config("spark.sql.catalog." + catalogName + ".root", testRoot)
            .config("spark.sql.catalog." + catalogName + ".single_level_ns", "true")
            .getOrCreate();
    this.tableName = "show_indexes_test_" + UUID.randomUUID().toString().replace("-", "");
    this.fullTable = this.catalogName + ".default." + this.tableName;
    this.tableDir =
        FileSystems.getDefault().getPath(testRoot, this.tableName + ".lance").toString();
  }

  @AfterEach
  public void tearDown() throws IOException {
    if (spark != null) {
      spark.close();
    }
  }

  private void prepareDataset() {
    spark.sql(String.format("create table %s (id int, text string) using lance;", fullTable));
    // First insert to create initial fragments
    spark.sql(
        String.format(
            "insert into %s (id, text) values %s ;",
            fullTable,
            IntStream.range(0, 10)
                .boxed()
                .map(i -> String.format("(%d, 'text_%d')", i, i))
                .collect(Collectors.joining(","))));
    // Second insert to ensure multiple fragments
    spark.sql(
        String.format(
            "insert into %s (id, text) values %s ;",
            fullTable,
            IntStream.range(10, 20)
                .boxed()
                .map(i -> String.format("(%d, 'text_%d')", i, i))
                .collect(Collectors.joining(","))));
  }

  @Test
  public void testShowIndexes() {
    prepareDataset();

    // Create a B-tree index on id
    spark.sql(String.format("alter table %s create index test_index using btree (id)", fullTable));

    Dataset<Row> result = spark.sql(String.format("show indexes from %s", fullTable));

    Assertions.assertEquals(
        "StructType(StructField(name,StringType,true),StructField(fields,ArrayType(StringType,true),true),StructField(index_type,StringType,true),StructField(num_indexed_fragments,LongType,true),StructField(num_indexed_rows,LongType,true),StructField(num_unindexed_fragments,LongType,true),StructField(num_unindexed_rows,LongType,true))",
        result.schema().toString());

    List<Row> rows = result.collectAsList();
    Assertions.assertFalse(rows.isEmpty(), "Expected at least one index row");

    Row row = rows.get(0);

    // name should match created index
    Assertions.assertEquals("test_index", row.getString(0));

    // fields should contain column name "id"
    @SuppressWarnings("unchecked")
    List<String> fieldNames = row.getList(1);
    Assertions.assertTrue(fieldNames.contains("id"), "fields should contain column name 'id'");

    // index_type should be btree
    Assertions.assertEquals("btree", row.getString(2));

    // num_indexed_fragments should be at least 1
    long numIndexedFragments = row.getLong(3);
    Assertions.assertTrue(numIndexedFragments >= 1L, "num_indexed_fragments should be at least 1");

    // num_indexed_rows should be at least 1
    long numIndexedRows = row.getLong(4);
    Assertions.assertTrue(numIndexedRows >= 1L, "num_indexed_rows should be at least 1");
  }

  private static final int VEC_DIM = 32;
  private static final int VEC_ROWS_PER_INSERT = 80;
  private static final int VEC_INSERT_COUNT = 4;
  private static final int VEC_NUM_ROWS = VEC_ROWS_PER_INSERT * VEC_INSERT_COUNT;

  /**
   * Create a table with a fixed-size-list vector column and insert several fragments so a
   * distributed IVF index build produces multiple segments.
   */
  private void prepareVectorDataset() {
    spark.sql(
        String.format(
            "CREATE TABLE %s (id INT NOT NULL, vec ARRAY<FLOAT> NOT NULL) USING lance "
                + "TBLPROPERTIES ('vec.arrow.fixed-size-list.size' = '%d')",
            fullTable, VEC_DIM));

    Random random = new Random(42);
    int rowId = 0;
    for (int b = 0; b < VEC_INSERT_COUNT; b++) {
      List<String> values = new ArrayList<>();
      for (int r = 0; r < VEC_ROWS_PER_INSERT; r++) {
        StringBuilder arr = new StringBuilder("ARRAY(");
        for (int i = 0; i < VEC_DIM; i++) {
          if (i > 0) arr.append(", ");
          arr.append(String.format(Locale.ROOT, "CAST(%f AS FLOAT)", random.nextFloat()));
        }
        arr.append(")");
        values.add(String.format("(%d, %s)", rowId++, arr));
      }
      spark.sql(
          String.format("INSERT INTO %s (id, vec) VALUES %s", fullTable, String.join(",", values)));
    }
  }

  /**
   * SHOW INDEXES must work for vector (IVF_*) indexes.
   *
   * <p>This is the index class that regressed in production: the previous implementation called
   * {@code Dataset.getIndexStatistics}, which serializes the full IVF centroid matrix into a JSON
   * string. For large multi-segment indexes that JSON reached multiple GB and failed to deserialize
   * on the driver, making SHOW INDEXES unusable. The fixed implementation derives the summary from
   * {@code describeIndices()} plus cheap dataset counts and never materializes centroids, so this
   * command succeeds regardless of index scale.
   */
  @Test
  public void testShowIndexesOnVectorIndex() {
    prepareVectorDataset();

    // Distributed IVF_PQ build across the inserted fragments.
    spark.sql(
        String.format(
            "ALTER TABLE %s CREATE INDEX vec_idx USING IVF_PQ (vec) "
                + "WITH (num_partitions=4, num_sub_vectors=4)",
            fullTable));

    Dataset<Row> result = spark.sql(String.format("show indexes from %s", fullTable));

    List<Row> rows = result.collectAsList();
    Assertions.assertEquals(1, rows.size(), "Expected exactly one index row");

    Row row = rows.get(0);
    Assertions.assertEquals("vec_idx", row.getString(0));

    @SuppressWarnings("unchecked")
    List<String> fieldNames = row.getList(1);
    Assertions.assertTrue(fieldNames.contains("vec"), "fields should contain column name 'vec'");

    // Vector index type is surfaced either as the specific IVF_* subtype or the umbrella "vector".
    String indexType = row.getString(2);
    Assertions.assertTrue(
        indexType.startsWith("ivf") || indexType.equals("vector"),
        "index_type should be an IVF/vector type, got: " + indexType);

    // The index covers every fragment exactly once, so all rows are indexed and none are left over.
    long numIndexedFragments = row.getLong(3);
    long numIndexedRows = row.getLong(4);
    long numUnindexedFragments = row.getLong(5);
    long numUnindexedRows = row.getLong(6);

    Assertions.assertTrue(numIndexedFragments >= 1L, "num_indexed_fragments should be at least 1");
    Assertions.assertEquals(VEC_NUM_ROWS, numIndexedRows, "all rows should be indexed");
    Assertions.assertEquals(
        0L, numUnindexedFragments, "no fragments should be left unindexed after a full build");
    Assertions.assertEquals(0L, numUnindexedRows, "no rows should be left unindexed");
  }

  /**
   * The fragment/row breakdown must reflect newly appended, not-yet-indexed data. After adding a
   * fragment post-index-build, SHOW INDEXES should report the new fragment and its rows as
   * unindexed. This exercises the dataset-total based accounting introduced by the fix.
   */
  @Test
  public void testShowIndexesReportsUnindexedDataForVectorIndex() {
    prepareVectorDataset();

    spark.sql(
        String.format(
            "ALTER TABLE %s CREATE INDEX vec_idx USING IVF_PQ (vec) "
                + "WITH (num_partitions=4, num_sub_vectors=4)",
            fullTable));

    // Append one more fragment that the index does not cover yet.
    List<String> values = new ArrayList<>();
    Random random = new Random(7);
    for (int r = 0; r < VEC_ROWS_PER_INSERT; r++) {
      StringBuilder arr = new StringBuilder("ARRAY(");
      for (int i = 0; i < VEC_DIM; i++) {
        if (i > 0) arr.append(", ");
        arr.append(String.format(Locale.ROOT, "CAST(%f AS FLOAT)", random.nextFloat()));
      }
      arr.append(")");
      values.add(String.format("(%d, %s)", VEC_NUM_ROWS + r, arr));
    }
    spark.sql(
        String.format("INSERT INTO %s (id, vec) VALUES %s", fullTable, String.join(",", values)));

    Dataset<Row> result = spark.sql(String.format("show indexes from %s", fullTable));
    Row row = result.collectAsList().get(0);

    long numIndexedRows = row.getLong(4);
    long numUnindexedFragments = row.getLong(5);
    long numUnindexedRows = row.getLong(6);

    Assertions.assertEquals(
        VEC_NUM_ROWS, numIndexedRows, "only the original rows should be indexed");
    Assertions.assertTrue(
        numUnindexedFragments >= 1L, "the appended fragment should be reported as unindexed");
    Assertions.assertEquals(
        VEC_ROWS_PER_INSERT, numUnindexedRows, "the appended rows should be reported as unindexed");
  }
}
