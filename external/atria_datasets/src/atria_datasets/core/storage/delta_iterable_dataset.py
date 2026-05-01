import logging
import math

import torch.distributed
from deltalake import DeltaTable
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)


class DeltaIterableDataset(IterableDataset):
    def __init__(
        self,
        path: str,
        version: int | None = None,
        use_fixed_rank: bool = False,
        rank: int = None,
        num_ranks: int = None,
        batch_size: int = 32,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.version = version
        self.use_fixed_rank = use_fixed_rank
        self.rank = rank
        self.num_ranks = num_ranks
        self.path = path
        self.batch_size = batch_size
        self.storage_options = storage_options
        self.init_boundaries(path)

    def init_boundaries(self, path, init_start_end: bool = True):
        self.start = 0
        self.end = self.count()
        logger.debug(f"Dataset for path {path}. Count:{self.end}")

        if self.use_fixed_rank:
            if init_start_end:
                self.start, self.end = self.calc_boundaries(
                    self.start, self.end, self.rank, self.num_ranks
                )
                logger.debug(
                    f"Using fixed rank.  Current rank: {self.rank} "
                    f"World size: {self.num_ranks}"
                )
                logger.debug(f"Start: {self.start} End: {self.end}")
        elif torch.distributed.is_initialized():
            self.num_ranks = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            logger.debug(
                f"Detected DDP process. "
                f"Current rank: {self.rank} World size: {self.num_ranks}"
            )
            if init_start_end:
                self.start, self.end = self.calc_boundaries(
                    self.start, self.end, self.rank, self.num_ranks
                )
                logger.debug(
                    f"This rank will use the following "
                    f"set of rows: {self.start}-{self.end}"
                )
        else:
            self.num_ranks = 1
            self.rank = 1

    @staticmethod
    def calc_boundaries(start, end, rank, num_ranks):
        per_worker_data_count = int(math.ceil((end - start) / float(num_ranks)))
        new_start = start + rank * per_worker_data_count
        new_end = min(start + (rank + 1) * per_worker_data_count, end)
        return new_start, new_end

    def count(self):
        _delta_table = self.create_delta_table()
        _add_actions = _delta_table.get_add_actions().to_pandas()
        num_records = _add_actions["num_records"].sum()
        del _delta_table
        return num_records

    def create_delta_table(self):
        delta_table = DeltaTable(
            self.path, version=self.version, storage_options=self.storage_options
        )
        conf = delta_table.metadata().configuration
        if conf:
            deletion_vectors = conf.get("delta.enableDeletionVectors", None)
            if deletion_vectors == "true":
                raise Exception(
                    "Tables with enabled Deletion Vectors are not supported."
                )
        return delta_table

    def __len__(self):
        return int(self.end - self.start)


class DeltaDataset(DeltaIterableDataset):
    def calc_chunk_boundaries_for_current_worker(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self.start, self.end
        else:
            iter_start, iter_end = DeltaIterableDataset.calc_boundaries(
                self.start, self.end, worker_info.id, worker_info.num_workers
            )
        return iter_start, iter_end

    def process_data(self):
        _filter = None
        iter_start, iter_end = self.calc_chunk_boundaries_for_current_worker()
        if iter_end > 0 and iter_start >= 0:
            _filter = (pc.field(self.id_field) >= pc.scalar(iter_start)) & (
                pc.field(self.id_field) < pc.scalar(iter_end)
            )

        delta_table = self.create_delta_table()
        scanner = delta_table.to_pyarrow_dataset().scanner(
            columns=self.arrow_fields, filter=_filter
        )
        for rb in scanner.to_reader():
            num_rows = rb.num_rows
            indexes = list(range(num_rows))
            if self.shuffle:
                random.shuffle(indexes)
            for i in indexes:
                item = rb.slice(offset=i, length=1).to_pylist()[0]
                item = DeltaIterableDataset.decode_and_transform_record(
                    item, self.field_specs
                )
                yield item
        del delta_table
