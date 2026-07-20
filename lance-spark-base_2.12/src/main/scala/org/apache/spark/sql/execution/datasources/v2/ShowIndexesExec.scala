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
package org.apache.spark.sql.execution.datasources.v2

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, GenericInternalRow}
import org.apache.spark.sql.catalyst.plans.logical.ShowIndexesOutputType
import org.apache.spark.sql.catalyst.util.GenericArrayData
import org.apache.spark.sql.connector.catalog.{Identifier, TableCatalog}
import org.apache.spark.unsafe.types.UTF8String
import org.lance.spark.LanceDataset
import org.lance.spark.utils.{FieldPathUtils, Utils}

import java.util.Locale

import scala.collection.JavaConverters._

/**
 * Physical execution of SHOW INDEXES for Lance datasets.
 *
 * This command lists all indexes defined on the underlying Lance table.
 *
 * The summary columns (index type and the indexed/unindexed fragment and row
 * counts) are derived entirely from `describeIndices()` plus cheap,
 * manifest-based dataset counts. In particular we deliberately avoid
 * `Dataset.getIndexStatistics`, which for vector (IVF_*) indexes serializes the
 * full centroid matrix into a JSON string. For large, multi-segment IVF indexes
 * that JSON can reach multiple gigabytes and fail to deserialize on the driver
 * (`Failed to deserialize from JSON` / Jackson `Decimal point not followed by a
 * digit`), which would make SHOW INDEXES unusable even though the index itself
 * is healthy. See the accompanying PR for details.
 */
case class ShowIndexesExec(
    catalog: TableCatalog,
    ident: Identifier) extends LeafV2CommandExec {

  override def output: Seq[Attribute] = ShowIndexesOutputType.SCHEMA

  override protected def run(): Seq[InternalRow] = {
    val lanceDataset = catalog.loadTable(ident) match {
      case ds: LanceDataset => ds
      case _ =>
        throw new UnsupportedOperationException("ShowIndexes only supports LanceDataset")
    }

    val readOptions = lanceDataset.readOptions()

    val dataset = Utils.openDatasetBuilder(readOptions).build()
    try {
      val indexes = dataset.describeIndices().asScala.toSeq
      val lanceSchema = dataset.getLanceSchema()

      // Dataset-level totals, computed once. These are cheap manifest-derived
      // counts (no data scan) and let us report the unindexed fragment/row
      // counts without touching the heavy per-index statistics path.
      val totalFragments = dataset.getFragments().size().toLong
      val totalRows = dataset.countRows()

      indexes.map { idx =>
        val fieldIds = idx.getFieldIds
        val fieldNamesArray =
          if (fieldIds == null) {
            null
          } else {
            val names = fieldIds.asScala.map { id =>
              val colName = Option(FieldPathUtils.pathByFieldId(lanceSchema, id))
                .getOrElse(id.toString)
              UTF8String.fromString(colName)
            }
            new GenericArrayData(names.toArray[AnyRef])
          }

        val name = idx.getName
        val indexTypeUtf8 =
          Option(idx.getIndexType)
            .map(t => UTF8String.fromString(t.toLowerCase(Locale.ROOT)))
            .orNull

        // Fragment coverage across all segments of this logical index. A
        // fragment that is covered by more than one delta segment is counted
        // once, matching the semantics of index_statistics.num_indexed_fragments.
        val segments = idx.getSegments.asScala
        val hasFragmentInfo = segments.nonEmpty && segments.forall(_.fragments().isPresent)

        val numIndexedRows: java.lang.Long = java.lang.Long.valueOf(idx.getRowsIndexed)

        if (hasFragmentInfo) {
          val indexedFragmentIds =
            segments.flatMap(_.fragments().get.asScala.map(_.intValue())).toSet
          val numIndexedFragments = indexedFragmentIds.size.toLong
          val numUnindexedFragments = math.max(0L, totalFragments - numIndexedFragments)
          val numUnindexedRows = math.max(0L, totalRows - idx.getRowsIndexed)

          new GenericInternalRow(Array[Any](
            UTF8String.fromString(name),
            fieldNamesArray,
            indexTypeUtf8,
            java.lang.Long.valueOf(numIndexedFragments),
            numIndexedRows,
            java.lang.Long.valueOf(numUnindexedFragments),
            java.lang.Long.valueOf(numUnindexedRows)))
        } else {
          // Legacy indices without a fragment bitmap: we can still report the
          // name, fields, type and indexed-row count, but cannot derive the
          // fragment-level breakdown, so those columns are null.
          new GenericInternalRow(Array[Any](
            UTF8String.fromString(name),
            fieldNamesArray,
            indexTypeUtf8,
            null,
            numIndexedRows,
            null,
            null))
        }
      }
    } finally {
      dataset.close()
    }
  }
}
