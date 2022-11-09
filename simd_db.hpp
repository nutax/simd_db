#include <cstdint>
#include <tuple>

namespace simd_db
{
    namespace detail
    {
        std::size_t constexpr round_multiple(std::size_t number, std::size_t multiple)
        {
            return ((number + multiple - 1) / multiple) * multiple;
        }
        std::size_t constexpr static_column_allocation_size(std::size_t vec_size, std::size_t capacity, std::size_t type_size)
        {
            return round_multiple(round_multiple(capacity * type_size, vec_size), type_size) / type_size;
        }

        template <std::size_t CACHE_LINE, std::size_t VEC_SIZE, std::size_t CAPACITY, typename T>
        struct static_column
        {
            using type = T;
            alignas(CACHE_LINE) T data[static_column_allocation_size(VEC_SIZE, CAPACITY, sizeof(T))];
            T &operator[](std::size_t index)
            {
                return data[index];
            }
        };
    };

    template <std::size_t CACHE_LINE, std::size_t VEC_SIZE, std::size_t CAPACITY, typename... Ts>
    class static_table
    {
        template <typename T>
        using column_template = detail::static_column<CACHE_LINE, VEC_SIZE, CAPACITY, T>;

        std::size_t _size = 0;
        std::tuple<column_template<Ts>...> _columns;

        template <std::size_t... Is>
        void destroy_impl(std::size_t i, std::index_sequence<Is...>)
        {
            _size--;
            (..., (std::get<Is>(_columns)[i] = std::get<Is>(_columns)[_size]));
        }

    public:
        template <std::size_t I>
        auto *column()
        {
            return std::get<I>(_columns).data;
        }

        template <std::size_t I>
        auto &create()
        {
            return std::get<I>(_columns)[_size];
        }

        std::size_t create()
        {
            return _size++;
        }

        void destroy(std::size_t i)
        {
            destroy_impl(i, std::make_index_sequence<sizeof...(Ts)>{});
        }

        std::size_t size() const
        {
            return _size;
        }

        template <std::size_t I>
        static constexpr std::size_t v_step()
        {
            return VEC_SIZE / sizeof(typename std::tuple_element<I, decltype(_columns)>::type::type);
        }
    };
};