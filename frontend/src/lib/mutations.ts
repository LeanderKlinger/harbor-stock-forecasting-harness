import { useMutation, useQueryClient } from "@tanstack/react-query";
import toast from "react-hot-toast";

import { deleteRuns, queryKeys } from "./api";
import type { RunIndexEntry } from "./types";

interface DeleteResponse {
  deleted: string[];
  not_found: string[];
  failed: string[];
}

interface DeleteContext {
  previousIndex: RunIndexEntry[] | undefined;
}

export function useDeleteRuns() {
  const queryClient = useQueryClient();
  return useMutation<DeleteResponse, Error, string[], DeleteContext>({
    mutationFn: (runIds: string[]) => deleteRuns(runIds),
    retry: false,
    onMutate: async (runIds) => {
      // Snapshot for rollback. Cancel in-flight refetches so they can't clobber
      // the optimistic state.
      await queryClient.cancelQueries({ queryKey: queryKeys.runsIndex });
      const previousIndex = queryClient.getQueryData<RunIndexEntry[]>(
        queryKeys.runsIndex,
      );
      if (previousIndex) {
        const target = new Set(runIds);
        queryClient.setQueryData<RunIndexEntry[]>(
          queryKeys.runsIndex,
          previousIndex.filter((e) => !target.has(e.run_id)),
        );
      }
      return { previousIndex };
    },
    onError: (err, _runIds, context) => {
      if (context?.previousIndex !== undefined) {
        queryClient.setQueryData(queryKeys.runsIndex, context.previousIndex);
      }
      toast.error(`Delete failed: ${err.message}`);
    },
    onSuccess: (data) => {
      const { deleted, not_found, failed } = data;
      if (failed.length === 0 && not_found.length === 0) {
        toast.success(`Deleted ${deleted.length} run${deleted.length === 1 ? "" : "s"}`);
      } else {
        toast.error(
          `Deleted ${deleted.length}; ${failed.length} failed, ${not_found.length} not found`,
        );
      }
    },
    onSettled: () => {
      // Re-sync against server truth regardless of outcome — covers partial
      // failures where the optimistic state diverged from reality.
      queryClient.invalidateQueries({ queryKey: queryKeys.runsIndex });
    },
  });
}
